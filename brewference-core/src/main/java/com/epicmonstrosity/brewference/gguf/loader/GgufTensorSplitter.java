package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;
import com.epicmonstrosity.brewference.tensor.MappedF16Tensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;

/**
 * Utilities for splitting fused GGUF tensors into their logical parts.
 * <p>
 * Some architectures (Phi-3, several llama-arch conversions, some Qwen exports) store the
 * attention projections as a single fused tensor {@code blk.N.attn_qkv.weight} instead of the
 * separate {@code attn_q.weight} / {@code attn_k.weight} / {@code attn_v.weight} tensors.
 */
public final class GgufTensorSplitter {
    private GgufTensorSplitter() {
    }

    /** The three projections extracted from a fused QKV tensor. */
    public record Qkv<T>(T q, T k, T v) { }

    /** The two projections stored by Phi-3 as one gate/up tensor. */
    public record GateUp<T>(T gate, T up) { }

    public static GateUp<QuantizedTensor> splitFusedGateUp(
            final QuantizedTensor fused,
            final int projectionElements,
            final String tensorName) {
        validateGateUp(fused, projectionElements, tensorName);
        return new GateUp<>(sliceQuantized(fused, 0, projectionElements, tensorName),
                sliceQuantized(fused, projectionElements, projectionElements, tensorName));
    }

    /**
     * Splits a fused quantized QKV tensor into its query, key-value parts.
     *
     * @param fused      the fused quantized tensor
     * @param qElements  number of elements belonging to Q ({@code queryAttentionWidth * inDim})
     * @param kElements  number of elements belonging to K ({@code keyValueDim * inDim})
     * @param vElements  number of elements belonging to V ({@code keyValueDim * inDim})
     * @param tensorName tensor name, used for diagnostics only
     * @return the three separated projections
     * @throws IllegalArgumentException if the sizes do not add up or are not block aligned
     */
    public static Qkv<QuantizedTensor> splitFusedQkv(final QuantizedTensor fused,
                                                     final int qElements,
                                                     final int kElements,
                                                     final int vElements,
                                                     final String tensorName) {
        if (fused == null)
            throw new IllegalArgumentException("Fused QKV tensor is null: %s"
                    .formatted(tensorName));

        validateSizes(fused.elementCount(), qElements, kElements, vElements, tensorName);
        validateBlockAlignment(fused.blockSize(), qElements, kElements, vElements, tensorName);
        return new Qkv<>(
                sliceQuantized(fused, 0, qElements, tensorName),
                sliceQuantized(fused, qElements, kElements, tensorName),
                sliceQuantized(fused, qElements + kElements, vElements, tensorName));
    }

    /**
     * Splits a fused QKV bias (or any other unquantized fused QKV tensor) into its three parts.
     *
     * @param fused      the fused float tensor
     * @param qElements  number of elements belonging to Q ({@code queryAttentionWidth})
     * @param kElements  number of elements belonging to K ({@code keyValueDim})
     * @param vElements  number of elements belonging to V ({@code keyValueDim})
     * @param tensorName tensor name, used for diagnostics only
     * @return the three separated bias vectors
     */
    public static Qkv<FloatTensor> splitFusedQkv(
            final FloatTensor fused,
            final int qElements,
            final int kElements,
            final int vElements,
            final String tensorName) {
        if (fused == null) {
            throw new IllegalArgumentException("Fused QKV tensor is null: %s"
                    .formatted(tensorName));
        }
        validateSizes(fused.elementCount(), qElements, kElements, vElements, tensorName);
        return new Qkv<>(
                sliceFloatTensor(fused, 0, qElements),
                sliceFloatTensor(fused, qElements, kElements),
                sliceFloatTensor(fused, qElements + kElements, vElements));
    }

    private static QuantizedTensor sliceQuantized(final QuantizedTensor src,
                                                  final int elementOffset,
                                                  final int elementCount,
                                                  final String tensorName) {
        final QuantizedTensor mapped = src.mappedSlice(elementOffset, elementCount);
        if (mapped != null) return mapped;
        throw new IllegalArgumentException("Quantized tensor cannot be sliced as a contiguous mapped view: %s"
                .formatted(tensorName));
    }

    private static FloatTensor sliceFloatTensor(final FloatTensor src,
                                                final int elementOffset,
                                                final int elementCount) {
        if (src instanceof final MappedF32Tensor mapped) {
            return new MappedF32Tensor(
                    mapped.data().asSlice(Math.multiplyExact((long) elementOffset, Float.BYTES),
                            Math.multiplyExact((long) elementCount, Float.BYTES)), elementCount);
        }
        if (src instanceof final MappedF16Tensor mapped) {
            return new MappedF16Tensor(
                    mapped.data().asSlice(Math.multiplyExact((long) elementOffset, Short.BYTES),
                            Math.multiplyExact((long) elementCount, Short.BYTES)), elementCount);
        }
        final float[] out = new float[elementCount];
        for (int i = 0; i < elementCount; i++) {
            out[i] = src.get(elementOffset + i);
        }
        return new FloatArrayTensor(out);
    }

    private static void validateSizes(final long total,
                                      final int qElements,
                                      final int kElements,
                                      final int vElements,
                                      final String tensorName) {
        if (qElements <= 0 || kElements <= 0 || vElements <= 0) {
            throw new IllegalArgumentException("Non-positive QKV split size for %s: q=%d k=%d v=%d"
                    .formatted(tensorName, qElements, kElements, vElements));
        }
        final long splitElements = (long) qElements + kElements + vElements;
        if (splitElements != total) {
            throw new IllegalArgumentException("Fused QKV size mismatch for %s: tensor has %d elements but q+k+v=%d"
                    .formatted(tensorName, total, splitElements));
        }
    }

    private static void validateGateUp(final QuantizedTensor fused,
                                       final int projectionElements,
                                       final String tensorName) {
        if (fused == null) {
            throw new IllegalArgumentException("Fused gate/up tensor is null: " + tensorName);
        }
        if (projectionElements <= 0 || fused.elementCount() != (long) projectionElements * 2) {
            throw new IllegalArgumentException("Fused gate/up size mismatch for " + tensorName +
                    ": tensor has " + fused.elementCount() + " elements, projection=" + projectionElements);
        }
        if (projectionElements % fused.blockSize() != 0) {
            throw new IllegalArgumentException("Fused gate/up tensor %s cannot be split on a block boundary (block=%d): projection=%d"
                    .formatted(tensorName, fused.blockSize(), projectionElements));
        }
    }

    private static void validateBlockAlignment(final int blockSize,
                                               final int qElements,
                                               final int kElements,
                                               final int vElements,
                                               final String tensorName) {
        if (qElements % blockSize != 0 || kElements % blockSize != 0 || vElements % blockSize != 0) {
            throw new IllegalArgumentException("Fused QKV tensor %s cannot be split on a block boundary (block=%d): q=%d k=%d v=%d"
                    .formatted(tensorName, blockSize, qElements, kElements, vElements));
        }
    }
}
