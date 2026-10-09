package com.epicmonstrosity.brewference.gguf.data;

import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.transformer.rope.LongRopeScaling;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public class QuantizedWeights {
    public QuantizedTensor tokenEmbeddingTable;
    public QuantizedTensor wq, wk, wv, wo;
    public QuantizedTensor w1, w2, w3;
    public QuantizedTensor classifier;

    // Bias slots accept any GGML type; quantized payloads dequantize lazily on read.
    public Tensor qBias;
    public Tensor kBias;
    public Tensor vBias;

    // Norm slots accept any GGML type; quantized payloads dequantize lazily on read.
    public Tensor rmsAttWeight;
    public Tensor rmsFfnWeight;
    public Tensor rmsFinalWeight;

    public Tensor postAttWeight;
    public Tensor postFfnWeight;

    public Tensor rmsKWeight;
    public Tensor rmsQWeight;

    // RoPE factor slots accept any GGML type; quantized payloads dequantize lazily on read.
    public float[] freqCisReal;
    public float[] freqCisImag;

    public LongRopeScaling ropeScaling;
    public Tensor ropeFactorsLong;
    public Tensor ropeFactorsShort;

    /**
     * Per-layer views of the same weights carried by the flat composites above.
     * <p>
     * For architectures whose layers are not shape-uniform — for example hybrid SSM / full-attention
     * models where {@code attn_q} is gated on some layers and {@code attn_qkv} is a fused linear-attention
     * projection on others — a flat composite cannot be addressed with {@code layer * dim * width} because
     * the width differs per layer. Those architectures must read the {@code *Layer} arrays instead.
     * <p>
     * Every array is nullable: it is null when the GGUF does not carry the corresponding tensor, and the
     * flat composites are unchanged so uniform architectures keep using them.
     */
    public QuantizedTensor[] wqLayer, wkLayer, wvLayer, woLayer, w1Layer, w2Layer, w3Layer;
    public Tensor[] qBiasLayer, kBiasLayer, vBiasLayer;
    public Tensor[] rmsAttLayer, rmsFfnLayer, postAttLayer, postFfnLayer, rmsQLayer, rmsKLayer;

    // Gated Delta Net (Qwen3.5 linear-attention blocks)
    public QuantizedTensor[] attnQkvLayer;
    public QuantizedTensor[] attnGateLayer;
    public QuantizedTensor[] ssmAlphaLayer;
    public QuantizedTensor[] ssmBetaLayer;
    public QuantizedTensor[] ssmOutLayer;
    public Tensor[] ssmALayer;
    public Tensor[] ssmConv1dLayer;
    public Tensor[] ssmDtBiasLayer;
    public Tensor[] ssmNormLayer;

    private final Map<String, Tensor> tensors = new LinkedHashMap<>();

    public void putTensor(final String name, final Tensor tensor) {
        tensors.put(Objects.requireNonNull(name, "name"), Objects.requireNonNull(tensor, "tensor"));
    }

    public Tensor getTensor(final String name) {
        return tensors.get(name);
    }

    public Map<String, Tensor> tensors() {
        return Collections.unmodifiableMap(tensors);
    }

    /**
     * Reads a row of a tensor into a float array, dequantizing lazily when the tensor carries a
     * quantized payload.
     *
     * @param q The Tensor holding the row data (quantized payloads are dequantized element by element).
     * @param rowOffset The starting offset within the tensor, indicating the first
     *                  element of the row to read.
     * @param n The total number of elements to read.
     * @param out The output array where the floating-point values are written.
     */
    public void dequantizeRow(final Tensor q, final int rowOffset, final int n, final float[] out) {
        if (q == null || out == null || rowOffset < 0 || n < 0 ||
                (long) rowOffset + n > q.elementCount() || out.length < n) {
            throw new IllegalArgumentException("Invalid quantized row range");
        }
        for (int i = 0; i < n; i++) {
            out[i] = q.value(rowOffset + i);
        }
    }
}
