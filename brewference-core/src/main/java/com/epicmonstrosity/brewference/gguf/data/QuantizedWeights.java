package com.epicmonstrosity.brewference.gguf.data;

import com.epicmonstrosity.brewference.tensor.FloatTensor;
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

    // Bias is float
    public FloatTensor qBias;
    public FloatTensor kBias;
    public FloatTensor vBias;

    // Norms are always float
    public FloatTensor rmsAttWeight;
    public FloatTensor rmsFfnWeight;
    public FloatTensor rmsFinalWeight;

    public FloatTensor postAttWeight;
    public FloatTensor postFfnWeight;

    public FloatTensor rmsKWeight;
    public FloatTensor rmsQWeight;

    // RoPE is float
    public float[] freqCisReal;
    public float[] freqCisImag;

    public LongRopeScaling ropeScaling;
    public FloatTensor ropeFactorsLong;
    public FloatTensor ropeFactorsShort;

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
     * Dequantizes a row of quantized data from a QuantizedTensor and writes the resulting floating-point
     * values into an output array.
     * <p>
     * Unlike other transfomers (llama.cpp), this dequant then GEMM approach is simpler but bloats the memory footprint.
     *
     * @param q The QuantizedTensor containing the quantized data and the corresponding scale factors.
     * @param rowOffset The starting offset within the QuantizedTensor's data array, indicating the first
     *                  element of the row to dequantize.
     * @param n The total number of elements to dequantize.
     * @param out The output array where the dequantized floating-point values are written.
     */
    public void dequantizeRow(final QuantizedTensor q, final int rowOffset, final int n, final float[] out) {
        if (q == null || out == null || rowOffset < 0 || n < 0 ||
                (long) rowOffset + n > q.elementCount() || out.length < n) {
            throw new IllegalArgumentException("Invalid quantized row range");
        }
        for (int i = 0; i < n; i++) {
            out[i] = q.value(rowOffset + i);
        }
    }
}
