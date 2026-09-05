package com.epicmonstrosity.brewference.gguf;

import com.epicmonstrosity.brewference.gguf.loader.GgufTensorSplitter;
import com.epicmonstrosity.brewference.tensor.CompositeFloatTensor;
import com.epicmonstrosity.brewference.tensor.CompositeQuantizedTensor;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;
import com.epicmonstrosity.brewference.tensor.MappedF16Tensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.transformer.math.Kernels;
import com.epicmonstrosity.brewference.transformer.math.Linear;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MappedTensorSupportTest {
    private static final int Q8_BLOCK_SIZE = 32;
    private static final int Q8_BLOCK_BYTES = Short.BYTES + Q8_BLOCK_SIZE;
    private static final int Q4_BLOCK_BYTES = Short.BYTES + Q8_BLOCK_SIZE / 2;

    private static final ValueLayout.OfShort LITTLE_ENDIAN_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LITTLE_ENDIAN_FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void quantizedSegmentTensorQ8ProvidesIndexedValuesAndScales() {
        final byte[] bytes = q8Bytes(1.5f, (byte) 7, -0.25f, (byte) -3);
        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q8_0, 64);

        assertEquals(64, tensor.elementCount());
        assertEquals(2, tensor.blockCount());
        assertEquals(1.5f, tensor.scale(0), 0.001f);
        assertEquals(-0.25f, tensor.scale(1), 0.001f);
        assertEquals(7, tensor.quantizedValue(0));
        assertEquals(-3, tensor.quantizedValue(32));

        final QuantizedSegmentTensor slice = tensor.mappedSlice(32, 32);
        assertEquals(32, slice.elementCount());
        assertEquals(-0.25f, slice.scale(0), 0.001f);
        assertEquals(-3, slice.quantizedValue(0));
    }

    @Test
    void quantizedSegmentTensorQ4DecodesPackedSignedNibblesAndProvidesMappedSlices() {
        final byte[] bytes = q4Bytes(0.5f, -0.25f);
        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q4_0, 64);

        assertEquals(64, tensor.elementCount());
        assertEquals(2, tensor.blockCount());
        assertEquals(0.5f, tensor.scale(0), 0.001f);
        assertEquals(-0.25f, tensor.scale(1), 0.001f);
        assertEquals(-8, tensor.quantizedValue(0));
        assertEquals(7, tensor.quantizedValue(31));
        assertEquals(-8, tensor.quantizedValue(32));

        final QuantizedSegmentTensor slice = tensor.mappedSlice(32, 32);
        bytes[Q4_BLOCK_BYTES + Short.BYTES] = (byte) 0x0f;
        assertEquals(7, slice.quantizedValue(0));
        assertThrows(IllegalArgumentException.class, () -> tensor.mappedSlice(1, 32));
    }

    @Test
    void quantizedSegmentTensorQ4UsesGgmlLowThenHighNibbleOrdering() {
        final byte[] bytes = new byte[Q4_BLOCK_BYTES];
        bytes[Short.BYTES + 1] = (byte) 0xf1;
        final QuantizedSegmentTensor tensor = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q4_0, 32);

        // this project's Q4_0 layout: elements 0-15 read the low nibble of data byte i,
        // elements 16-31 read the high nibble of data byte i - 16
        assertEquals(-7, tensor.quantizedValue(1));
        assertEquals(7, tensor.quantizedValue(17));
    }

    @Test
    void mappedF16AndF32DecodeLittleEndianValues() {
        final byte[] f32Bytes = new byte[Float.BYTES * 2];
        final MemorySegment f32Data = MemorySegment.ofArray(f32Bytes);
        f32Data.setAtIndex(LITTLE_ENDIAN_FLOAT, 0, 1.25f);
        f32Data.setAtIndex(LITTLE_ENDIAN_FLOAT, 1, -2.5f);
        final FloatTensor f32 = new MappedF32Tensor(f32Data.asReadOnly(), 2);

        final byte[] f16Bytes = new byte[Short.BYTES * 2];
        final MemorySegment f16Data = MemorySegment.ofArray(f16Bytes);
        f16Data.setAtIndex(LITTLE_ENDIAN_SHORT, 0, Float.floatToFloat16(0.5f));
        f16Data.setAtIndex(LITTLE_ENDIAN_SHORT, 1, Float.floatToFloat16(-3.0f));
        final FloatTensor f16 = new MappedF16Tensor(f16Data.asReadOnly(), 2);

        assertArrayEquals(new float[]{1.25f, -2.5f}, f32.toArray());
        assertArrayEquals(new float[]{0.5f, -3.0f}, f16.toArray());
    }

    @Test
    void normalizationAndBiasReadMappedFloatTensorsByIndex() {
        final byte[] bytes = new byte[Float.BYTES * 6];
        final MemorySegment data = MemorySegment.ofArray(bytes);
        final float[] values = {2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f};
        for (int i = 0; i < values.length; i++) {
            data.setAtIndex(LITTLE_ENDIAN_FLOAT, i, values[i]);
        }
        final FloatTensor tensor = new MappedF32Tensor(data.asReadOnly(), values.length);

        final float[] normalized = new float[2];
        Kernels.rmsNorm(normalized, new float[]{3.0f, 4.0f}, tensor, 1, 2, 0.0f);
        assertArrayEquals(new float[]{2.5455844f, 4.525483f}, normalized, 0.0001f);

        final float[] heads = {3.0f, 4.0f, 6.0f, 8.0f};
        Kernels.headWiseRmsNorm(heads, tensor, 0, 2, 2, 0.0f);
        assertArrayEquals(new float[]{1.6970563f, 3.3941126f, 1.6970563f, 3.3941126f}, heads, 0.0001f);

        final float[] biased = {1.0f, 2.0f, 3.0f};
        Kernels.addBias(biased, tensor, 3, 3);
        assertArrayEquals(new float[]{11.0f, 22.0f, 33.0f}, biased, 0.0001f);
    }

    @Test
    void linearMatmulQ8SegmentMatchesReference() {
        final byte[] bytes = q8Bytes(0.5f, (byte) 2, -0.25f, (byte) 4);
        final QuantizedTensor weights = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q8_0, 64);
        final float[] input = new float[32];
        for (int i = 0; i < input.length; i++) {
            input[i] = i - 10;
        }
        final float[] output = new float[2];

        Linear.matmul(output, input, weights, 0, 32, 2);

        float expectedRow0 = 0.0f;
        float expectedRow1 = 0.0f;
        for (int i = 0; i < input.length; i++) {
            expectedRow0 += 2 * 0.5f * input[i];
            expectedRow1 += 4 * -0.25f * input[i];
        }
        assertEquals(expectedRow0, output[0], 0.0001f);
        assertEquals(expectedRow1, output[1], 0.0001f);
    }

    @Test
    void linearMatmulMatchesQ4ReferenceAcrossCompositeViews() {
        final QuantizedTensor[] parts = new QuantizedTensor[]{
                segment(q4Bytes(0.5f), GgmlType.Q4_0, 32),
                segment(q4Bytes(-0.25f), GgmlType.Q4_0, 32),
                segment(q4Bytes(0.25f), GgmlType.Q4_0, 32),
                segment(q4Bytes(-0.5f), GgmlType.Q4_0, 32)
        };
        final CompositeQuantizedTensor weights = new CompositeQuantizedTensor(parts);
        final float[] input = new float[64];
        for (int i = 0; i < input.length; i++) input[i] = i - 13.5f;
        final float[] output = new float[1];

        Linear.matmul(output, input, weights, 64, 64, 1);

        float expected = 0.0f;
        for (int i = 0; i < input.length; i++) {
            // row spans parts 3 and 4; the q4Bytes helper fills both nibbles with pair,
            // so under this project's layout the value is local<16 ? local-8 : local-24
            final int local = i % 32;
            final float scale = i < 32 ? 0.25f : -0.5f;
            expected += (local < 16 ? local - 8 : local - 24) * scale * input[i];
        }
        assertEquals(expected, output[0], 0.0001f);
    }

    @Test
    void linearMatmulQ4SegmentDirectBlockPathAcrossRows() {
        final byte[] bytes = q4Bytes(0.5f, -0.25f, 0.75f, -0.5f);
        final QuantizedTensor weights = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q4_0, 128);
        final float[] input = new float[32];
        for (int i = 0; i < input.length; i++) input[i] = (i - 11) * 0.25f;
        final float[] output = new float[2];

        Linear.matmul(output, input, weights, 64, 32, 2);

        for (int row = 0; row < output.length; row++) {
            float expected = 0.0f;
            for (int column = 0; column < input.length; column++) {
                expected += weights.value(64L + (long) row * input.length + column) * input[column];
            }
            assertEquals(expected, output[row], 0.0001f);
        }
    }

    @Test
    void linearMatmulPackedSegmentLoadsPreserveSignedQ8Values() {
        final byte[] values = {
                -128, -64, -17, -1, 0, 1, 15, 63,
                127, 42, -42, 99, -99, 8, -8, 3,
                -3, 100, -100, 32, -32, 7, -7, 2,
                -2, 96, -96, 11, -11, 55, -55, 4
        };
        final byte[] bytes = q8Bytes(0.75f, (byte) 0);
        System.arraycopy(values, 0, bytes, Short.BYTES, values.length);
        final QuantizedTensor weights = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q8_0, 32);
        final float[] input = new float[32];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i - 13) * 0.25f;
        }
        final float[] output = new float[1];

        Linear.matmul(output, input, weights, 0, 32, 1);

        float expected = 0.0f;
        for (int i = 0; i < input.length; i++) {
            expected += values[i] * 0.75f * input[i];
        }
        assertEquals(expected, output[0], 0.0001f);
    }

    @Test
    void linearMatmulVectorizedSegmentLoadsMatchReferenceAcrossBlocks() {
        final byte[] bytes = q8Bytes(0.5f, (byte) 0, -0.25f, (byte) 0);
        for (int i = 0; i < 64; i++) {
            final byte value = (byte) (i * 17 - 101);
            bytes[Short.BYTES + i % Q8_BLOCK_SIZE + (i / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES] = value;
        }
        final QuantizedTensor weights = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q8_0, 64);
        final float[] input = new float[32];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i - 7) * 0.125f;
        }
        final float[] output = new float[2];

        Linear.matmul(output, input, weights, 0, 32, 2);

        for (int row = 0; row < output.length; row++) {
            float expected = 0.0f;
            for (int i = 0; i < input.length; i++) {
                final byte value = (byte) ((row * 32 + i) * 17 - 101);
                final float scale = row == 0 ? 0.5f : -0.25f;
                expected += value * scale * input[i];
            }
            assertEquals(expected, output[row], 0.0001f);
        }
    }

    @Test
    void fusedMappedSlicesRemainViewsOfTheSourcePayload() {
        final byte[] q8Bytes = q8Bytes(0.5f, (byte) 1, 1.0f, (byte) 2, 1.5f, (byte) 3);
        final QuantizedTensor fusedQ8 = new QuantizedSegmentTensor(
                MemorySegment.ofArray(q8Bytes).asReadOnly(), GgmlType.Q8_0, 96);
        final GgufTensorSplitter.Qkv<QuantizedTensor> qkv =
                GgufTensorSplitter.splitFusedQkv(fusedQ8, 32, 32, 32, "qkv");

        assertEquals(32, qkv.k().elementCount());
        assertEquals(1.0f, qkv.k().scale(0), 0.001f);
        q8Bytes[Q8_BLOCK_BYTES + Short.BYTES] = 9;
        assertEquals(9.0f, qkv.k().value(0), 0.0001f);

        final byte[] f32Bytes = new byte[Float.BYTES * 3];
        final MemorySegment source = MemorySegment.ofArray(f32Bytes);
        source.setAtIndex(LITTLE_ENDIAN_FLOAT, 0, 1.0f);
        source.setAtIndex(LITTLE_ENDIAN_FLOAT, 1, 2.0f);
        source.setAtIndex(LITTLE_ENDIAN_FLOAT, 2, 3.0f);
        final FloatTensor fusedFloat = new MappedF32Tensor(source.asReadOnly(), 3);
        final GgufTensorSplitter.Qkv<FloatTensor> floats =
                GgufTensorSplitter.splitFusedQkv(fusedFloat, 1, 1, 1, "bias");

        source.setAtIndex(LITTLE_ENDIAN_FLOAT, 1, 7.0f);
        assertEquals(7.0f, floats.k().get(0));
    }

    @Test
    void quantizedFusedSlicesRequireBlockAlignedBoundaries() {
        final QuantizedTensor fused = new QuantizedSegmentTensor(
                MemorySegment.ofArray(q8Bytes(1.0f, (byte) 1, 1.0f, (byte) 1, 1.0f, (byte) 1)).asReadOnly(),
                GgmlType.Q8_0, 96);

        assertThrows(IllegalArgumentException.class,
                () -> GgufTensorSplitter.splitFusedQkv(fused, 31, 33, 32, "unaligned"));
    }

    @Test
    void fusedQ4MappedSlicesRemainViewsOfTheSourcePayload() {
        final byte[] bytes = q4Bytes(0.5f, 1.0f, 1.5f);
        final QuantizedTensor fused = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q4_0, 96);

        final GgufTensorSplitter.Qkv<QuantizedTensor> qkv =
                GgufTensorSplitter.splitFusedQkv(fused, 32, 32, 32, "q4qkv");

        assertEquals(1.0f, qkv.k().scale(0), 0.001f);
        bytes[Q4_BLOCK_BYTES + Short.BYTES] = (byte) 0x0f;
        assertEquals(7.0f, qkv.k().value(0), 0.001f);
        assertThrows(IllegalArgumentException.class,
                () -> GgufTensorSplitter.splitFusedQkv(fused, 31, 33, 32, "unaligned"));
    }

    @Test
    void compositeViewsJoinPartsWithoutMaterializingPayloads() {
        final byte[] firstBytes = q8Bytes(0.5f, (byte) 1);
        final byte[] secondBytes = q8Bytes(1.5f, (byte) 2);
        final CompositeQuantizedTensor q8 = new CompositeQuantizedTensor(
                segment(firstBytes, GgmlType.Q8_0, 32),
                segment(secondBytes, GgmlType.Q8_0, 32));
        final CompositeFloatTensor floats = new CompositeFloatTensor(
                new FloatArrayTensor(new float[]{1.0f}),
                new FloatArrayTensor(new float[]{2.0f, 3.0f}));

        assertEquals(64, q8.elementCount());
        assertEquals(1.5f, q8.scale(1), 0.001f);
        secondBytes[Short.BYTES] = 9;
        assertEquals(13.5f, q8.value(32), 0.0001f);
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, floats.toArray());
        assertThrows(IndexOutOfBoundsException.class, () -> q8.value(64));
        assertThrows(IndexOutOfBoundsException.class, () -> floats.get(-1));
    }

    @Test
    void linearMatmulResolvesOneMappedCompositePartBeforeReadingBlocks() {
        final CompositeQuantizedTensor weights = new CompositeQuantizedTensor(
                segment(q8Bytes(0.5f, (byte) 1), GgmlType.Q8_0, 32),
                segment(q8Bytes(2.0f, (byte) 3), GgmlType.Q8_0, 32));
        final float[] output = new float[1];
        final float[] input = new float[32];
        Arrays.fill(input, 1.0f);

        Linear.matmul(output, input, weights, 32, 32, 1);

        assertTrue(weights.mappedSlice(32, 32) instanceof QuantizedSegmentTensor slice
                && slice.data().byteSize() == Q8_BLOCK_BYTES);
        assertEquals(192.0f, output[0]);
        assertThrows(IllegalArgumentException.class, () -> weights.mappedSlice(1, 32));
        assertNull(weights.mappedSlice(0, 64));
    }

    private static QuantizedSegmentTensor segment(final byte[] bytes, final GgmlType type, final long elements) {
        return new QuantizedSegmentTensor(MemorySegment.ofArray(bytes).asReadOnly(), type, elements);
    }

    private static byte[] q8Bytes(final float... scalesAndValues) {
        final byte[] bytes = new byte[scalesAndValues.length / 2 * Q8_BLOCK_BYTES];
        final MemorySegment data = MemorySegment.ofArray(bytes);
        for (int block = 0; block < scalesAndValues.length / 2; block++) {
            final int offset = block * Q8_BLOCK_BYTES;
            data.set(LITTLE_ENDIAN_SHORT, offset, Float.floatToFloat16(scalesAndValues[block * 2]));
            for (int value = 0; value < Q8_BLOCK_SIZE; value++) {
                bytes[offset + Short.BYTES + value] = (byte) scalesAndValues[block * 2 + 1];
            }
        }
        return bytes;
    }

    private static byte[] q4Bytes(final float... scales) {
        final byte[] bytes = new byte[scales.length * Q4_BLOCK_BYTES];
        final MemorySegment data = MemorySegment.ofArray(bytes);
        for (int block = 0; block < scales.length; block++) {
            final int offset = block * Q4_BLOCK_BYTES;
            data.set(LITTLE_ENDIAN_SHORT, offset, Float.floatToFloat16(scales[block]));
            for (int pair = 0; pair < Q8_BLOCK_SIZE / 2; pair++) {
                // low and high nibbles are identical: both decode to (pair - 8)
                bytes[offset + Short.BYTES + pair] = (byte) (pair | pair << 4);
            }
        }
        return bytes;
    }
}
