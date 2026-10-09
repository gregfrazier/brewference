package com.epicmonstrosity.brewference.gguf.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.tensor.CompositeTensor;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.MappedF16Tensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.transformer.math.Kernels;
import com.epicmonstrosity.brewference.transformer.math.Linear;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * A float payload and a quantized payload must be interchangeable in both weight slot kinds.
 * <p>
 * Matmul slots go through {@link GgufWeightsLoader#quantizedSlot} (an F32/F16 payload is
 * re-expressed as a {@link QuantizedSegmentTensor} over the same memory); element-wise slots go
 * through {@link GgufWeightsLoader#elementWise} and dequantize lazily through
 * {@link Tensor#value(long)}. These tests build small payloads by hand and pin both directions:
 * quantized weight in a kernel that used to demand floats, float payload in a matmul slot, a
 * composite that mixes the two, and BF16 end to end.
 */
class TensorInterchangeTest {
    private static final int ELEMENTS = 64;
    private static final int ROWS = 4;
    private static final int COLUMNS = 8;
    private static final float EPSILON = 1e-6f;

    /** Integral weights keep a scale-1.0 Q8_0 block exact: value == scale * byte == the float. */
    private static final float[] INTEGER_WEIGHTS = {
            1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
            2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
            9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f,
            1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f,
            7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f, 13.0f, -14.0f,
            15.0f, -16.0f, 17.0f, -18.0f, 19.0f, -20.0f, 21.0f, -22.0f,
    };

    private static final float[] INPUT = {
            1.5f, -2.0f, 0.25f, 3.5f, -0.75f, 4.25f, -3.125f, 0.5f,
    };

    private static float[] inputs(final int size) {
        final float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = INPUT[i % INPUT.length] * (1.0f + (i % 5) * 0.125f);
        }
        return values;
    }

    private static MappedF32Tensor f32Tensor(final float[] values) {
        return new MappedF32Tensor(MemorySegment.ofArray(values).asReadOnly(), values.length);
    }

    private static MappedF16Tensor f16Tensor(final float[] values) {
        final short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++)
            bits[i] = Float.floatToFloat16(values[i]);
        return new MappedF16Tensor(MemorySegment.ofArray(bits).asReadOnly(), bits.length);
    }

    /** Builds a Q8_0 payload with scale 1.0 per block so every stored byte is the value itself. */
    private static QuantizedSegmentTensor q8Tensor(final float[] integralValues) {
        final int blocks = (integralValues.length + 31) / 32;
        final byte[] payload = new byte[blocks * 34];
        final short scaleBits = Float.floatToFloat16(1.0f);
        for (int block = 0; block < blocks; block++) {
            payload[block * 34] = (byte) (scaleBits & 0xFF);
            payload[block * 34 + 1] = (byte) ((scaleBits >>> 8) & 0xFF);
            for (int i = 0; i < 32; i++)
                payload[block * 34 + Short.BYTES + i] = (byte) (int) integralValues[block * 32 + i];
        }
        return new QuantizedSegmentTensor(MemorySegment.ofArray(payload).asReadOnly(),
                GgmlType.Q8_0, integralValues.length);
    }

    private static float[] referenceMatmul(final Tensor weights, final int rows, final int columns) {
        final float[] out = new float[rows];
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            for (int column = 0; column < columns; column++)
                sum += weights.value((long) row * columns + column) * INPUT[column];
            out[row] = sum;
        }
        return out;
    }

    @Test
    void rmsNormWithQuantizedWeightMatchesFloatWeight() {
        final float[] input = inputs(ELEMENTS);
        final Tensor floatWeight = new FloatArrayTensor(INTEGER_WEIGHTS);
        final Tensor quantizedWeight = q8Tensor(INTEGER_WEIGHTS);

        final float[] expected = new float[ELEMENTS];
        Kernels.rmsNorm(expected, input, floatWeight, 0, ELEMENTS, EPSILON);

        final float[] actual = new float[ELEMENTS];
        Kernels.rmsNorm(actual, input, quantizedWeight, 0, ELEMENTS, EPSILON);

        for (int i = 0; i < ELEMENTS; i++)
            assertEquals(expected[i], actual[i], 1e-6f, "rmsNorm drift at " + i);
    }

    @Test
    void addBiasWithQuantizedBiasMatchesFloatBias() {
        final float[] base = inputs(ELEMENTS);
        final Tensor floatBias = new FloatArrayTensor(INTEGER_WEIGHTS);
        final Tensor quantizedBias = q8Tensor(INTEGER_WEIGHTS);

        final float[] expected = base.clone();
        Kernels.addBias(expected, floatBias, 0, ELEMENTS);

        final float[] actual = base.clone();
        Kernels.addBias(actual, quantizedBias, 0, ELEMENTS);

        for (int i = 0; i < ELEMENTS; i++)
            assertEquals(expected[i], actual[i], 1e-6f, "addBias drift at " + i);
    }

    @Test
    void quantizedSlotAcceptsMappedF32AndF16InAClassicSlot() {
        final float[] weightValues = new float[ROWS * COLUMNS];
        for (int i = 0; i < weightValues.length; i++)
            weightValues[i] = INTEGER_WEIGHTS[i % INTEGER_WEIGHTS.length] * 0.25f;

        final QuantizedTensor f32Slot = assertInstanceOf(QuantizedSegmentTensor.class,
                GgufWeightsLoader.quantizedSlot("blk.0.ffn_gate.weight", f32Tensor(weightValues)));
        final QuantizedTensor f16Slot = assertInstanceOf(QuantizedSegmentTensor.class,
                GgufWeightsLoader.quantizedSlot("blk.0.ffn_gate.weight", f16Tensor(weightValues)));

        assertEquals(GgmlType.F32, ((QuantizedSegmentTensor) f32Slot).ggmlType());
        assertEquals(GgmlType.F16, ((QuantizedSegmentTensor) f16Slot).ggmlType());
        assertEquals(1, f32Slot.blockSize());
        assertEquals(1, f16Slot.blockSize());

        final float[] expectedF32 = referenceMatmul(f32Slot, ROWS, COLUMNS);
        final float[] outF32 = new float[ROWS];
        Linear.matmul(outF32, INPUT, f32Slot, 0, COLUMNS, ROWS);
        for (int row = 0; row < ROWS; row++)
            assertEquals(expectedF32[row], outF32[row], 1e-5f, "F32 matmul drift at row " + row);

        final float[] expectedF16 = referenceMatmul(f16Slot, ROWS, COLUMNS);
        final float[] outF16 = new float[ROWS];
        Linear.matmul(outF16, INPUT, f16Slot, 0, COLUMNS, ROWS);
        for (int row = 0; row < ROWS; row++)
            assertEquals(expectedF16[row], outF16[row], 1e-4f, "F16 matmul drift at row " + row);
    }

    @Test
    void elementWiseSlotAcceptsQuantizedPayload() {
        final QuantizedSegmentTensor q8 = q8Tensor(INTEGER_WEIGHTS);

        final Tensor slot = GgufWeightsLoader.elementWise("blk.0.ssm_norm.weight", q8);

        assertSame(q8, slot, "element-wise slots must not copy or dequantize eagerly");
        assertEquals(ELEMENTS, slot.elementCount());
        for (int i = 0; i < ELEMENTS; i++)
            assertEquals(INTEGER_WEIGHTS[i], slot.value(i), 1e-6f, "dequantized value drift at " + i);
    }

    @Test
    void compositeTensorMixesFloatAndQuantizedParts() {
        final float[] floatPart = {1.5f, -2.25f, 3.75f};
        final float[] quantizedPart = new float[32];
        System.arraycopy(INTEGER_WEIGHTS, 0, quantizedPart, 0, 32);

        final CompositeTensor composite = new CompositeTensor(f32Tensor(floatPart), q8Tensor(quantizedPart));

        assertEquals(floatPart.length + quantizedPart.length, composite.elementCount());
        assertEquals(floatPart[0], composite.value(0), "first index reads the float part");
        assertEquals(floatPart[2], composite.value(2), "last float index");
        assertEquals(quantizedPart[0], composite.value(floatPart.length), 1e-6f, "first index of the quantized part");
        assertEquals(quantizedPart[31], composite.value(composite.elementCount() - 1), 1e-6f, "last index");

        final float[] expected = new float[floatPart.length + quantizedPart.length];
        System.arraycopy(floatPart, 0, expected, 0, floatPart.length);
        System.arraycopy(quantizedPart, 0, expected, floatPart.length, quantizedPart.length);
        assertArrayEquals(expected, composite.toArray(), 1e-6f);
    }

    @Test
    void bf16PayloadRoundTripsAndMatmulMatches() {
        final float[] weightValues = new float[ROWS * COLUMNS];
        for (int i = 0; i < weightValues.length; i++)
            weightValues[i] = INTEGER_WEIGHTS[i % INTEGER_WEIGHTS.length] * 0.5f;

        final short[] bits = new short[weightValues.length];
        for (int i = 0; i < weightValues.length; i++)
            bits[i] = (short) (Float.floatToIntBits(weightValues[i]) >>> 16);

        final QuantizedSegmentTensor bf16 = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bits).asReadOnly(), GgmlType.BF16, bits.length);

        assertEquals(GgmlType.BF16, bf16.ggmlType());
        assertEquals(1, bf16.blockSize());
        for (int i = 0; i < bits.length; i++) {
            final float rounded = Float.intBitsToFloat(((int) bits[i]) << 16);
            assertEquals(rounded, bf16.value(i), "BF16 round trip at " + i);
            assertEquals(weightValues[i], bf16.value(i), "value lost precision at " + i);
        }

        final float[] expected = referenceMatmul(bf16, ROWS, COLUMNS);
        final float[] out = new float[ROWS];
        Linear.matmul(out, INPUT, bf16, 0, COLUMNS, ROWS);
        for (int row = 0; row < ROWS; row++)
            assertEquals(expected[row], out[row], 1e-2f, "BF16 matmul drift at row " + row);
    }
}
