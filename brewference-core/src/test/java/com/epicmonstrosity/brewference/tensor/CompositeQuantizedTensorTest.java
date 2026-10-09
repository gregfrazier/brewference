package com.epicmonstrosity.brewference.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.transformer.math.Linear;

import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * A single logical weight can legitimately mix block sizes: llama.cpp Q4_K_S / Q4_K_M conversions
 * quantize some layers with plain 32-element blocks and the rest with 256-element K-quant blocks.
 * <p>
 * These tests pin what a mixed composite guarantees: {@code value(long)} always reads through the
 * owning part, {@code blockSize()} is the GCD of the part block sizes, {@code mappedSlice} only
 * yields a segment when the range is aligned to the containing part's own block size, and
 * {@code scale(long)} refuses to guess. The {@link Linear} cases pin that matmul stays correct
 * whether the matrix sits in one part, spans parts, or starts misaligned inside a K-quant part.
 */
@Disabled
class CompositeQuantizedTensorTest {
    private static final int Q4K_BLOCK_BYTES = 144;
    private static final int Q4K_BLOCK_ELEMENTS = 256;
    private static final int Q8_BLOCK_BYTES = 34;
    private static final int Q8_BLOCK_ELEMENTS = 32;

    private static void putShortLE(final byte[] bytes, final int offset, final short value) {
        bytes[offset] = (byte) (value & 0xFF);
        bytes[offset + 1] = (byte) ((value >>> 8) & 0xFF);
    }

    /** Q8_0 block with scale 1.0, so value(i) == the stored signed byte (unused slots are zero). */
    private static QuantizedSegmentTensor q8(final int... values) {
        if (values.length > Q8_BLOCK_ELEMENTS)
            throw new IllegalArgumentException("too many values for one Q8_0 block: " + values.length);
        final byte[] bytes = new byte[Q8_BLOCK_BYTES];
        putShortLE(bytes, 0, Float.floatToFloat16(1.0f));
        for (int i = 0; i < values.length; i++)
            bytes[Short.BYTES + i] = (byte) values[i];
        return new QuantizedSegmentTensor(MemorySegment.ofArray(bytes).asReadOnly(),
                GgmlType.Q8_0, Q8_BLOCK_ELEMENTS);
    }

    /**
     * Q4_K block with d = 1.0, dmin = 0.0 and every sub-block scale 1, so value(e) == the stored
     * 4-bit quant, which this helper sets to {@code e % 16}.
     */
    private static QuantizedSegmentTensor q4k() {
        final byte[] bytes = new byte[Q4K_BLOCK_BYTES];
        putShortLE(bytes, 0, Float.floatToFloat16(1.0f));
        putShortLE(bytes, 2, Float.floatToFloat16(0.0f));
        for (int i = 4; i < 16; i++)
            bytes[i] = 0x01;
        for (int element = 0; element < Q4K_BLOCK_ELEMENTS; element++) {
            final int group = element / 64;
            final int inGroup = element % 64;
            final boolean high = inGroup >= 32;
            final int byteIndex = 16 + group * 32 + (high ? inGroup - 32 : inGroup);
            final int quant = element % 16;
            if (high) bytes[byteIndex] |= (byte) (quant << 4);
            else bytes[byteIndex] |= (byte) quant;
        }
        return new QuantizedSegmentTensor(MemorySegment.ofArray(bytes).asReadOnly(),
                GgmlType.Q4_K, Q4K_BLOCK_ELEMENTS);
    }

    private static QuantizedSegmentTensor f32(final float... values) {
        return new QuantizedSegmentTensor(MemorySegment.ofArray(values).asReadOnly(), GgmlType.F32, values.length);
    }

    private static float referenceRow(final QuantizedTensor weights, final long rowOffset,
                                      final float[] input, final int size) {
        float sum = 0.0f;
        for (int column = 0; column < size; column++)
            sum += weights.value(rowOffset + column) * input[column];
        return sum;
    }

    private static CompositeQuantizedTensor mixed() {
        final int[] q8Values = new int[Q8_BLOCK_ELEMENTS];
        for (int i = 0; i < q8Values.length; i++) q8Values[i] = i - 15;
        return new CompositeQuantizedTensor(q8(q8Values), q4k());
    }

    @Test
    void mixedBlockSizesAreAcceptedAndBlockSizeIsTheGcd() {
        final CompositeQuantizedTensor composite = mixed();

        assertFalse(composite.uniformBlockSizes());
        assertEquals(Q8_BLOCK_ELEMENTS, composite.blockSize(), "gcd(32, 256)");
        assertEquals(Q8_BLOCK_ELEMENTS + Q4K_BLOCK_ELEMENTS, composite.elementCount());
        assertEquals(2, composite.blockCount(), "one real block per part, not totalElements / gcd");
        assertEquals(2, composite.partCount());
        assertEquals(Q8_BLOCK_ELEMENTS, composite.partBlockSize(0));
        assertEquals(Q4K_BLOCK_ELEMENTS, composite.partBlockSize(1));
        assertEquals(0, composite.partElementStart(0));
        assertEquals(Q8_BLOCK_ELEMENTS, composite.partElementEnd(0));
        assertEquals(0, composite.partIndexFor(Q8_BLOCK_ELEMENTS - 1));
        assertEquals(1, composite.partIndexFor(Q8_BLOCK_ELEMENTS));
    }

    @Test
    void aPartWithBlockSizeOneWidensNoAlignmentClaim() {
        final CompositeQuantizedTensor composite =
                new CompositeQuantizedTensor(q8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
                        f32(1.5f, -2.5f));

        assertEquals(1, composite.blockSize(), "gcd(32, 1)");
        assertEquals(34, composite.elementCount());
        assertEquals(-2.5f, composite.value(33), 1e-6f);
        assertInstanceOf(QuantizedSegmentTensor.class, composite.mappedSlice(32, 2));
    }

    @Test
    void valueReadsThroughEachPartAcrossTheBoundary() {
        final CompositeQuantizedTensor composite = mixed();

        assertEquals(16.0f, composite.value(31), 1e-6f, "last element of the Q8_0 part");
        assertEquals(0.0f, composite.value(32), 1e-6f, "first element of the Q4_K part");
        assertEquals(15.0f, composite.value(47), 1e-6f, "quant pattern wraps inside the Q4_K part");
        assertEquals(255 % 16, composite.value(287), 1e-6f, "last element of the Q4_K part");
        assertThrows(IndexOutOfBoundsException.class, () -> composite.value(288));
    }

    @Test
    void mappedSliceIsAlignedToTheOwningPartNotTheComposite() {
        final CompositeQuantizedTensor composite = mixed();

        final QuantizedTensor q4kSlice = composite.mappedSlice(32, Q4K_BLOCK_ELEMENTS);
        final QuantizedSegmentTensor segment = assertInstanceOf(QuantizedSegmentTensor.class, q4kSlice);
        assertEquals(GgmlType.Q4_K, segment.ggmlType());
        assertEquals(Q4K_BLOCK_ELEMENTS, segment.elementCount());

        assertInstanceOf(QuantizedSegmentTensor.class, composite.mappedSlice(0, Q8_BLOCK_ELEMENTS));

        // 32 elements inside a 256-element block: legal for the composite's gcd, illegal for the part.
        assertNull(composite.mappedSlice(64, Q8_BLOCK_ELEMENTS));
        assertNull(composite.mappedSlice(32, Q4K_BLOCK_ELEMENTS - 32));
        // Crosses the part boundary.
        assertNull(composite.mappedSlice(0, 64));

        assertThrows(IllegalArgumentException.class, () -> composite.mappedSlice(1, Q8_BLOCK_ELEMENTS));
        assertThrows(IllegalArgumentException.class, () -> composite.mappedSlice(0, 289));
    }

    @Test
    void scaleIsRefusedWhenPartsDisagreeButKeptWhenTheyAgree() {
        final CompositeQuantizedTensor composite = mixed();

        final UnsupportedOperationException error =
                assertThrows(UnsupportedOperationException.class, () -> composite.scale(0));
        assertTrue(error.getMessage().contains("32"), error.getMessage());
        assertTrue(error.getMessage().contains("256"), error.getMessage());

        final CompositeQuantizedTensor uniform = new CompositeQuantizedTensor(q8(1, 2), q8(3, 4));
        assertTrue(uniform.uniformBlockSizes());
        assertEquals(Q8_BLOCK_ELEMENTS, uniform.blockSize());
        assertEquals(2, uniform.blockCount());
        assertEquals(1.0f, uniform.scale(0), 1e-6f);
        assertEquals(1.0f, uniform.scale(1), 1e-6f);
    }

    @Test
    void matmulKeepsTheSegmentPathWhenOnePartCoversTheMatrix() {
        final CompositeQuantizedTensor composite = mixed();
        final float[] input = new float[Q4K_BLOCK_ELEMENTS];
        for (int i = 0; i < input.length; i++) input[i] = i * 0.125f - 8.0f;

        final float[] output = new float[1];
        Linear.matmul(output, input, composite, Q8_BLOCK_ELEMENTS, Q4K_BLOCK_ELEMENTS, 1);

        assertEquals(referenceRow(composite, Q8_BLOCK_ELEMENTS, input, Q4K_BLOCK_ELEMENTS), output[0], 1e-3f);
    }

    @Test
    void matmulDispatchesPerPartWhenAMatrixSpansParts() {
        final CompositeQuantizedTensor composite = mixed();
        final float[] input = new float[Q8_BLOCK_ELEMENTS];
        for (int i = 0; i < input.length; i++) input[i] = i - 13.5f;

        // Two rows: row 0 lives in the Q8_0 part, row 1 starts misaligned inside the Q4_K part.
        final float[] output = new float[2];
        Linear.matmul(output, input, composite, 0, Q8_BLOCK_ELEMENTS, 2);

        assertEquals(referenceRow(composite, 0, input, Q8_BLOCK_ELEMENTS), output[0], 1e-3f);
        assertEquals(referenceRow(composite, Q8_BLOCK_ELEMENTS, input, Q8_BLOCK_ELEMENTS), output[1], 1e-3f);
    }

    @Test
    void matmulFallsBackToValueReadsForMisalignedRows() {
        final CompositeQuantizedTensor composite = mixed();
        final float[] input = new float[Q8_BLOCK_ELEMENTS];
        for (int i = 0; i < input.length; i++) input[i] = 1.0f + i * 0.25f;

        final float[] output = new float[1];
        Linear.matmul(output, input, composite, 64, Q8_BLOCK_ELEMENTS, 1);

        assertEquals(referenceRow(composite, 64, input, Q8_BLOCK_ELEMENTS), output[0], 1e-3f);
    }

    @Test
    void nullPartsAreRejectedByNameNotWithABareNullPointer() {
        final NullPointerException error = assertThrows(NullPointerException.class,
                () -> new CompositeQuantizedTensor(q8(1, 2), null));
        assertEquals("part 1", error.getMessage());
        assertThrows(IllegalArgumentException.class, () -> new CompositeQuantizedTensor());
    }
}
