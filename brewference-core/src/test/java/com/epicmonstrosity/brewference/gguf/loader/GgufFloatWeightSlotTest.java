package com.epicmonstrosity.brewference.gguf.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.MappedF16Tensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.transformer.math.Linear;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Weight slots that a checkpoint may store either block-quantized or as plain F32/F16 floats.
 * <p>
 * Conversions disagree about which tensors are worth quantizing: some Qwen3.5 exports keep the
 * gated-delta-net family ({@code ssm_alpha.weight}, {@code ssm_beta.weight},
 * {@code attn_gate.weight}, the hybrid {@code attn_qkv.weight}) in F32 while quantizing the
 * attention and FFN bulk. {@link GgufWeightsLoader#quantizedSlot} re-expresses such a
 * payload as a {@link QuantizedSegmentTensor} over the same mapped memory so the matmul keeps its
 * vectorized segment path. These tests pin that contract: zero copy, type preserved, values
 * unchanged, matmul identical to a float loop, and the strict slots still strict.
 */
class GgufFloatWeightSlotTest {
    private static final int ROWS = 3;
    private static final int COLUMNS = 4;

    private static final float[] WEIGHT_VALUES = {
            0.5f, -1.25f, 3.0f, 0.0f,
            2.5f, 0.75f, -0.125f, 4.0f,
            -3.5f, 1.0f, 0.25f, -2.0f,
    };

    private static final float[] INPUT = {1.5f, -2.0f, 0.25f, 3.5f};

    private static MappedF32Tensor f32Tensor(final float[] values) {
        return new MappedF32Tensor(MemorySegment.ofArray(values).asReadOnly(), values.length);
    }

    private static MappedF16Tensor f16Tensor(final float[] values) {
        final short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++)
            bits[i] = Float.floatToFloat16(values[i]);
        return new MappedF16Tensor(MemorySegment.ofArray(bits).asReadOnly(), bits.length);
    }

    private static float[] referenceMatmul(final QuantizedTensor weights) {
        final float[] out = new float[ROWS];
        for (int row = 0; row < ROWS; row++) {
            float sum = 0.0f;
            for (int column = 0; column < COLUMNS; column++)
                sum += weights.value((long) row * COLUMNS + column) * INPUT[column];
            out[row] = sum;
        }
        return out;
    }

    @Test
    void f32SlotBecomesAnF32SegmentOverTheSameMemory() {
        final MappedF32Tensor source = f32Tensor(WEIGHT_VALUES);

        final QuantizedTensor adapted = GgufWeightsLoader.quantizedSlot("blk.0.ssm_alpha.weight", source);

        final QuantizedSegmentTensor segment =
                assertInstanceOf(QuantizedSegmentTensor.class, adapted);
        assertEquals(GgmlType.F32, segment.ggmlType());
        assertEquals(WEIGHT_VALUES.length, segment.elementCount());
        assertEquals(1, segment.blockSize(), "unquantized payloads are one-element blocks");
        // Zero copy: the adapted tensor reads the very same mapped region as the source view.
        assertSame(source.data(), segment.segment());
        for (int i = 0; i < WEIGHT_VALUES.length; i++)
            assertEquals(WEIGHT_VALUES[i], segment.value(i), "value drift for F32 slot at " + i);
    }

    @Test
    void f16SlotBecomesAnF16SegmentOverTheSameMemory() {
        final MappedF16Tensor source = f16Tensor(WEIGHT_VALUES);

        final QuantizedSegmentTensor segment = assertInstanceOf(QuantizedSegmentTensor.class,
                GgufWeightsLoader.quantizedSlot("blk.1.ssm_beta.weight", source));

        assertEquals(GgmlType.F16, segment.ggmlType());
        assertEquals(WEIGHT_VALUES.length, segment.elementCount());
        assertSame(source.data(), segment.segment());
        for (int i = 0; i < WEIGHT_VALUES.length; i++) {
            final float expected = Float.float16ToFloat(Float.floatToFloat16(WEIGHT_VALUES[i]));
            assertEquals(expected, segment.value(i), "value drift for F16 slot at " + i);
        }
    }

    @Test
    void quantizedSlotPassesThroughUnchanged() {
        final byte[] payload = new byte[34 * 2]; // two Q8_0 blocks
        final QuantizedSegmentTensor quantized =
                new QuantizedSegmentTensor(MemorySegment.ofArray(payload).asReadOnly(), GgmlType.Q8_0, 64);

        assertSame(quantized, GgufWeightsLoader.quantizedSlot("blk.0.ssm_out.weight", quantized));
    }

    @Test
    void wrappedF32SliceStaysOnTheSegmentMatmulPath() {
        final QuantizedSegmentTensor segment = assertInstanceOf(QuantizedSegmentTensor.class,
                GgufWeightsLoader.quantizedSlot("blk.0.ssm_alpha.weight", f32Tensor(WEIGHT_VALUES)));

        // Linear.matmul fast-paths only QuantizedSegmentTensor operands; a wrapped float weight
        // must keep slicing into one, or it would fall into per-element value() reads.
        final QuantizedTensor sliced = segment.mappedSlice(COLUMNS, COLUMNS);
        assertInstanceOf(QuantizedSegmentTensor.class, sliced);
        assertEquals(GgmlType.F32, ((QuantizedSegmentTensor) sliced).ggmlType());
        assertEquals(COLUMNS, sliced.elementCount());
    }

    @Test
    void wrappedF32WeightMatmulMatchesFloatLoop() {
        final QuantizedTensor weights =
                GgufWeightsLoader.quantizedSlot("blk.0.ssm_alpha.weight", f32Tensor(WEIGHT_VALUES));
        final float[] expected = referenceMatmul(weights);

        final float[] out = new float[ROWS];
        Linear.matmul(out, INPUT, weights, 0, COLUMNS, ROWS);

        for (int row = 0; row < ROWS; row++)
            assertEquals(expected[row], out[row], 1e-5f, "F32 matmul drift at row " + row);
    }

    @Test
    void wrappedF16WeightMatmulMatchesFloatLoop() {
        final QuantizedTensor weights =
                GgufWeightsLoader.quantizedSlot("blk.0.ssm_beta.weight", f16Tensor(WEIGHT_VALUES));
        final float[] expected = referenceMatmul(weights);

        final float[] out = new float[ROWS];
        Linear.matmul(out, INPUT, weights, 0, COLUMNS, ROWS);

        for (int row = 0; row < ROWS; row++)
            assertEquals(expected[row], out[row], 1e-4f, "F16 matmul drift at row " + row);
    }

    @Test
    void wrappedFloatWeightReportsNoScaleInsteadOfAnInventedOne() {
        final QuantizedSegmentTensor segment = assertInstanceOf(QuantizedSegmentTensor.class,
                GgufWeightsLoader.quantizedSlot("blk.0.ssm_alpha.weight", f32Tensor(WEIGHT_VALUES)));

        final UnsupportedOperationException error =
                assertThrows(UnsupportedOperationException.class, () -> segment.scale(0));
        assertTrue(error.getMessage().contains("F32"), error.getMessage());
    }

    @Test
    void heapFloatWeightIsRejectedWithTheTensorName() {
        final FloatArrayTensor heap = new FloatArrayTensor(WEIGHT_VALUES);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> GgufWeightsLoader.quantizedSlot("blk.0.ssm_alpha.weight", heap));
        assertTrue(error.getMessage().contains("blk.0.ssm_alpha.weight"), error.getMessage());
        assertTrue(error.getMessage().contains("FloatArrayTensor"), error.getMessage());
    }
}
