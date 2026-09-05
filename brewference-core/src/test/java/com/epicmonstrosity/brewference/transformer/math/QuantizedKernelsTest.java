package com.epicmonstrosity.brewference.transformer.math;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizedKernelsTest {
    private static final ValueLayout.OfShort LE_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void ggmlTypeMetadataAndByteSizeCalculations() {
        assertEquals(4, GgmlType.F32.getTypeSize());
        assertEquals(1, GgmlType.F32.getBlockSize());
        assertEquals(400, GgmlType.F32.byteSizeFor(100));

        assertEquals(2, GgmlType.F16.getTypeSize());
        assertEquals(1, GgmlType.F16.getBlockSize());
        assertEquals(200, GgmlType.F16.byteSizeFor(100));

        assertEquals(18, GgmlType.Q4_0.getTypeSize());
        assertEquals(32, GgmlType.Q4_0.getBlockSize());
        assertEquals(36, GgmlType.Q4_0.byteSizeFor(64));

        assertEquals(34, GgmlType.Q8_0.getTypeSize());
        assertEquals(32, GgmlType.Q8_0.getBlockSize());
        assertEquals(68, GgmlType.Q8_0.byteSizeFor(64));

        assertEquals(84, GgmlType.Q2_K.getTypeSize());
        assertEquals(256, GgmlType.Q2_K.getBlockSize());
        assertEquals(84, GgmlType.Q2_K.byteSizeFor(256));

        assertEquals(110, GgmlType.Q3_K.getTypeSize());
        assertEquals(256, GgmlType.Q3_K.getBlockSize());
        assertEquals(110, GgmlType.Q3_K.byteSizeFor(256));

        assertEquals(144, GgmlType.Q4_K.getTypeSize());
        assertEquals(256, GgmlType.Q4_K.getBlockSize());
        assertEquals(144, GgmlType.Q4_K.byteSizeFor(256));

        assertEquals(176, GgmlType.Q5_K.getTypeSize());
        assertEquals(256, GgmlType.Q5_K.getBlockSize());
        assertEquals(176, GgmlType.Q5_K.byteSizeFor(256));

        assertEquals(210, GgmlType.Q6_K.getTypeSize());
        assertEquals(256, GgmlType.Q6_K.getBlockSize());
        assertEquals(210, GgmlType.Q6_K.byteSizeFor(256));

        assertEquals(18, GgmlType.Q1_0.getTypeSize());
        assertEquals(128, GgmlType.Q1_0.getBlockSize());
        assertEquals(36, GgmlType.Q1_0.byteSizeFor(256));

        assertEquals(GgmlType.Q4_0, GgmlType.fromId(2));
        assertEquals(GgmlType.Q8_0, GgmlType.fromId(8));
        assertEquals(GgmlType.Q2_K, GgmlType.fromId(10));
        assertEquals(GgmlType.Q3_K, GgmlType.fromId(11));
        assertEquals(GgmlType.Q4_K, GgmlType.fromId(12));
        assertEquals(GgmlType.Q5_K, GgmlType.fromId(13));
        assertEquals(GgmlType.Q6_K, GgmlType.fromId(14));
        assertEquals(GgmlType.Q1_0, GgmlType.fromId(41));

        assertTrue(GgmlType.Q4_0.isQuantized());
        assertTrue(GgmlType.Q8_0.isQuantized());
        assertTrue(GgmlType.Q2_K.isQuantized());
        assertTrue(GgmlType.Q3_K.isQuantized());
        assertTrue(GgmlType.Q4_K.isQuantized());
        assertTrue(GgmlType.Q5_K.isQuantized());
        assertTrue(GgmlType.Q6_K.isQuantized());
        assertTrue(GgmlType.Q1_0.isQuantized());
        assertTrue(!GgmlType.F32.isQuantized());
    }

    @Test
    void quantizedSegmentTensorQ8DotMatchesScalarCalculation() {
        final int elements = 64;
        final byte[] bytes = new byte[34 * 2];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // Block 0: scale = 2.0f, all weights = 3
        segment.set(LE_SHORT, 0, Float.floatToFloat16(2.0f));
        for (int i = 0; i < 32; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) 3);
        }

        // Block 1: scale = -0.5f, all weights = -4
        segment.set(LE_SHORT, 34, Float.floatToFloat16(-0.5f));
        for (int i = 0; i < 32; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 34 + 2 + i, (byte) -4);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q8_0, elements);
        assertEquals(elements, tensor.elementCount());
        assertEquals(2, tensor.blockCount());
        assertEquals(2.0f, tensor.scale(0), 0.001f);
        assertEquals(-0.5f, tensor.scale(1), 0.001f);
        assertEquals(6.0f, tensor.value(0), 0.001f);
        assertEquals(2.0f, tensor.value(32), 0.001f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ8(segment.asReadOnly(), 0, input, 0, elements);
        // Block 0 sum: 32 * 3 * 2.0f = 192.0f
        // Block 1 sum: 32 * (-4) * (-0.5f) = 64.0f
        // Total = 256.0f
        assertEquals(256.0f, dotResult, 0.01f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(256.0f, output[0], 0.01f);
    }

    @Test
    void quantizedSegmentTensorQ1DotMatchesCalculation() {
        final int elements = 160;
        final byte[] bytes = new byte[18 * 2];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // Block 0: d = 2.0f, all bits set (0xFF) so every weight is +d = 2.0
        segment.set(LE_SHORT, 0, Float.floatToFloat16(2.0f));
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) 0xFF);
        }

        // Block 1: d = -1.0f, bytes 0x0F so bits 0-3 are set (weight d = -1.0)
        // and bits 4-7 are clear (weight -d = +1.0). Each byte contributes 4 * (-1) + 4 * (1) = 0.
        segment.set(LE_SHORT, 18, Float.floatToFloat16(-1.0f));
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 18 + 2 + i, (byte) 0x0F);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q1_0, 256);
        assertEquals(256, tensor.elementCount());
        assertEquals(2, tensor.blockCount());
        assertEquals(2.0f, tensor.scale(0), 0.001f);
        assertEquals(-1.0f, tensor.scale(1), 0.001f);
        assertEquals(2.0f, tensor.value(0), 0.001f);
        assertEquals(-1.0f, tensor.value(128), 0.001f);
        assertEquals(1.0f, tensor.value(133), 0.001f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        // Block 0: 128 * 2.0 = 256.0; block 1 remainder (32 elems): 4 bytes * (4 * -1 + 4 * 1) = 0.
        final float dotResult = QuantizedKernels.dotQ1(segment.asReadOnly(), 0, input, 0, elements);
        assertEquals(256.0f, dotResult, 0.01f);
    }

    @Test
    void quantizedSegmentTensorQ4DotMatchesCalculation() {
        final int elements = 64;
        final byte[] bytes = new byte[18 * 2];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // Block 0: scale = 1.0f, packed bytes = 0x77 (nibbles are 7, values are 7 - 8 = -1)
        segment.set(LE_SHORT, 0, Float.floatToFloat16(1.0f));
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) 0x77);
        }

        // Block 1: scale = 2.0f, packed bytes = 0x99 (nibbles are 9, values are 9 - 8 = 1)
        segment.set(LE_SHORT, 18, Float.floatToFloat16(2.0f));
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 18 + 2 + i, (byte) 0x99);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q4_0, elements);
        assertEquals(-1.0f, tensor.value(0), 0.001f);
        assertEquals(2.0f, tensor.value(32), 0.001f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ4(segment.asReadOnly(), 0, input, 0, elements);
        // Block 0: 32 * (-1) * 1.0f = -32.0f
        // Block 1: 32 * (1) * 2.0f = 64.0f
        // Total = 32.0f
        assertEquals(32.0f, dotResult, 0.01f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(32.0f, output[0], 0.01f);
    }

    @Test
    void quantizedSegmentTensorQ6KDotMatchesScalarCalculation() {
        final int elements = 256;
        final byte[] bytes = new byte[210];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // Super-block scale d = 0.5f at offset 208
        segment.set(LE_SHORT, 208, Float.floatToFloat16(0.5f));

        // Group scales: 16 int8 values at offset 192 (set all to 2)
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 192 + i, (byte) 2);
        }

        // ql nibbles at offset 0 (128 bytes): set all to 0x44 (nibble 4)
        for (int i = 0; i < 128; i++) {
            segment.set(ValueLayout.JAVA_BYTE, i, (byte) 0x44);
        }

        // qh bits at offset 128 (64 bytes): set all to 0x55 (bits 1 each)
        // high bits = 1 -> (nibble | (1 << 4)) = 4 | 16 = 20
        // q6 = 20 - 32 = -12
        for (int i = 0; i < 64; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 128 + i, (byte) 0x55);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q6_K, elements);
        assertEquals(256, tensor.elementCount());
        assertEquals(1, tensor.blockCount());

        // value = d * sc * q6 = 0.5 * 2 * (-12) = -12.0f
        assertEquals(-12.0f, tensor.value(0), 0.01f);
        assertEquals(-12.0f, tensor.value(128), 0.01f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ6K(segment.asReadOnly(), 0, input, 0, elements);
        // Total = 256 * (-12.0f) = -3072.0f
        assertEquals(-3072.0f, dotResult, 1.0f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(-3072.0f, output[0], 1.0f);
    }

    @Test
    void quantizedSegmentTensorSlicingComputesCorrectSubsegment() {
        final byte[] bytes = new byte[34 * 4];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        for (int b = 0; b < 4; b++) {
            segment.set(LE_SHORT, b * 34, Float.floatToFloat16(b + 1.0f));
            for (int i = 0; i < 32; i++) {
                segment.set(ValueLayout.JAVA_BYTE, b * 34 + 2 + i, (byte) (b + 1));
            }
        }

        final QuantizedSegmentTensor fullTensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q8_0, 128);
        final QuantizedSegmentTensor slice = fullTensor.slice(32, 64);

        assertEquals(64, slice.elementCount());
        assertEquals(2, slice.blockCount());
        assertEquals(2.0f, slice.scale(0), 0.001f);
        assertEquals(3.0f, slice.scale(1), 0.001f);
        assertEquals(4.0f, slice.value(0), 0.001f); // scale 2 * val 2 = 4
        assertEquals(9.0f, slice.value(32), 0.001f); // scale 3 * val 3 = 9

        assertThrows(IllegalArgumentException.class, () -> fullTensor.slice(1, 32));
        assertThrows(IllegalArgumentException.class, () -> fullTensor.slice(0, 33));
    }

    @Test
    void extensibleQuantizationKernelRegistry() {
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q8_0));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q4_0));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q6_K));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q4_K));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q5_K));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q2_K));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q3_K));
        assertNotNull(QuantizedKernels.getKernel(GgmlType.Q1_0));

        // Test registering a new custom kernel for Q4_1
//        QuantizedKernels.registerKernel(GgmlType.Q4_1, (seg, offset, in, inOffset, size) -> 42.0f);
//        assertEquals(42.0f, QuantizedKernels.getKernel(GgmlType.Q4_1).dot(null, 0, null, 0, 0));
    }

    @Test
    void quantizedSegmentTensorQ4KDotMatchesCalculation() {
        final int elements = 256;
        final byte[] bytes = new byte[144];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // d = 1.0f at offset 0, dmin = 0.5f at offset 2
        segment.set(LE_SHORT, 0, Float.floatToFloat16(1.0f));
        segment.set(LE_SHORT, 2, Float.floatToFloat16(0.5f));

        // scales at offset 4 (12 bytes):
        // sub-blocks 0..3: scale = 2, min = 1
        for (int j = 0; j < 4; j++) {
            segment.set(ValueLayout.JAVA_BYTE, 4 + j, (byte) 2); // scale
            segment.set(ValueLayout.JAVA_BYTE, 4 + 4 + j, (byte) 1); // min
        }
        // sub-blocks 4..7: low 4 bits = scale (2), high 4 bits = min (1), high 2 bits = 0
        for (int j = 0; j < 4; j++) {
            segment.set(ValueLayout.JAVA_BYTE, 4 + 8 + j, (byte) (2 | (1 << 4)));
        }

        // qs at offset 16 (128 bytes): set low and high nibbles to 3 (byte = 0x33)
        for (int i = 0; i < 128; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x33);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q4_K, elements);
        assertEquals(256, tensor.elementCount());
        assertEquals(1, tensor.blockCount());

        // value = d * sc * quant - dmin * min = 1.0 * 2 * 3 - 0.5 * 1 = 5.5f
        assertEquals(5.5f, tensor.value(0), 0.01f);
        assertEquals(5.5f, tensor.value(32), 0.01f);
        assertEquals(5.5f, tensor.value(128), 0.01f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ4K(segment.asReadOnly(), 0, input, 0, elements);
        // Total = 256 * 5.5f = 1408.0f
        assertEquals(1408.0f, dotResult, 1.0f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(1408.0f, output[0], 1.0f);
    }

    @Test
    void quantizedSegmentTensorQ5KDotMatchesCalculation() {
        final int elements = 256;
        final byte[] bytes = new byte[176];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // d = 1.0f at offset 0, dmin = 0.5f at offset 2
        segment.set(LE_SHORT, 0, Float.floatToFloat16(1.0f));
        segment.set(LE_SHORT, 2, Float.floatToFloat16(0.5f));

        // scales at offset 4 (12 bytes):
        for (int j = 0; j < 4; j++) {
            segment.set(ValueLayout.JAVA_BYTE, 4 + j, (byte) 2); // scale
            segment.set(ValueLayout.JAVA_BYTE, 4 + 4 + j, (byte) 1); // min
        }
        for (int j = 0; j < 4; j++) {
            segment.set(ValueLayout.JAVA_BYTE, 4 + 8 + j, (byte) (2 | (1 << 4)));
        }

        // qh at offset 16 (32 bytes): set all bits to 1 (0xFF) -> high bit = 1
        for (int i = 0; i < 32; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0xFF);
        }

        // qs at offset 48 (128 bytes): set low and high nibbles to 3 (byte = 0x33)
        // quant = 3 | (1 << 4) = 19
        for (int i = 0; i < 128; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 48 + i, (byte) 0x33);
        }

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q5_K, elements);
        assertEquals(256, tensor.elementCount());
        assertEquals(1, tensor.blockCount());

        // value = d * sc * quant - dmin * min = 1.0 * 2 * 19 - 0.5 * 1 = 37.5f
        assertEquals(37.5f, tensor.value(0), 0.01f);
        assertEquals(37.5f, tensor.value(32), 0.01f);
        assertEquals(37.5f, tensor.value(128), 0.01f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ5K(segment.asReadOnly(), 0, input, 0, elements);
        // Total = 256 * 37.5f = 9600.0f
        assertEquals(9600.0f, dotResult, 1.0f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(9600.0f, output[0], 1.0f);
    }

    @Test
    void quantizedSegmentTensorQ2KDotMatchesCalculation() {
        final int elements = 256;
        final byte[] bytes = new byte[84];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // scales at offset 0 (16 bytes): set all to 0x12 -> sc = 2, min = 1
        for (int i = 0; i < 16; i++) {
            segment.set(ValueLayout.JAVA_BYTE, i, (byte) 0x12);
        }

        // qs at offset 16 (64 bytes): set all to 0x55 -> 2-bit quants are 1
        for (int i = 0; i < 64; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x55);
        }

        // d = 1.0f at offset 80, dmin = 0.5f at offset 82
        segment.set(LE_SHORT, 80, Float.floatToFloat16(1.0f));
        segment.set(LE_SHORT, 82, Float.floatToFloat16(0.5f));

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q2_K, elements);
        assertEquals(256, tensor.elementCount());
        assertEquals(1, tensor.blockCount());

        // value = d * sc * q - dmin * min = 1.0 * 2 * 1 - 0.5 * 1 = 1.5f
        assertEquals(1.5f, tensor.value(0), 0.01f);
        assertEquals(1.5f, tensor.value(32), 0.01f);
        assertEquals(1.5f, tensor.value(128), 0.01f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ2K(segment.asReadOnly(), 0, input, 0, elements);
        // Total = 256 * 1.5f = 384.0f
        assertEquals(384.0f, dotResult, 1.0f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(384.0f, output[0], 1.0f);
    }

    @Test
    void quantizedSegmentTensorQ3KDotMatchesCalculation() {
        final int elements = 256;
        final byte[] bytes = new byte[110];
        final MemorySegment segment = MemorySegment.ofArray(bytes);

        // hm at offset 0 (32 bytes): set all to 0 -> (hm & m) = 0 -> hVal = 4
        for (int i = 0; i < 32; i++) {
            segment.set(ValueLayout.JAVA_BYTE, i, (byte) 0x00);
        }

        // qs at offset 32 (64 bytes): set all to 0xFF -> 2-bit quants are 3
        for (int i = 0; i < 64; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 32 + i, (byte) 0xFF);
        }

        // scales at offset 96 (12 bytes):
        // aux0 (bytes 0..3) = 0x22222222
        // aux1 (bytes 4..7) = 0x22222222
        // aux2 (bytes 8..11) = 0xAAAAAAAA
        // Reconstructs all 16 scales as 34 (meaning scale = (34 & 63) - 32 = 2)
        for (int i = 0; i < 8; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 96 + i, (byte) 0x22);
        }
        for (int i = 0; i < 4; i++) {
            segment.set(ValueLayout.JAVA_BYTE, 96 + 8 + i, (byte) 0xAA);
        }

        // d = 1.0f at offset 108
        segment.set(LE_SHORT, 108, Float.floatToFloat16(1.0f));

        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(segment.asReadOnly(), GgmlType.Q3_K, elements);
        assertEquals(256, tensor.elementCount());
        assertEquals(1, tensor.blockCount());

        // value = d * scale * (q - hVal) = 1.0 * 2 * (3 - 4) = -2.0f
        assertEquals(-2.0f, tensor.value(0), 0.01f);
        assertEquals(-2.0f, tensor.value(32), 0.01f);
        assertEquals(-2.0f, tensor.value(128), 0.01f);

        final float[] input = new float[elements];
        Arrays.fill(input, 1.0f);

        final float dotResult = QuantizedKernels.dotQ3K(segment.asReadOnly(), 0, input, 0, elements);
        // Total = 256 * (-2.0f) = -512.0f
        assertEquals(-512.0f, dotResult, 1.0f);

        final float[] output = new float[1];
        Linear.matmul(output, input, tensor, 0, elements, 1);
        assertEquals(-512.0f, output[0], 1.0f);
    }
}
