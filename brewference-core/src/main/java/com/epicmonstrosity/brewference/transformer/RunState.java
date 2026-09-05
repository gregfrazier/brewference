package com.epicmonstrosity.brewference.transformer;

import com.epicmonstrosity.brewference.transformer.cache.KvCache;

/**
 * Mutable state for the transformer forward pass. All buffers here are working memory for
 * activations and attention; their sizes depend only on model architecture (dim, kvDim,
 * hiddenDim, vocabSize, maxSeqLen), never on the weight quantization type.
 * <p>
 * Quantized weights (e.g. Q4_0) affect only weight storage: each row is dequantized to
 * native {@code float} before use (or consumed directly by quantized dot kernels whose inputs
 * and outputs are still FP32). A Q4_0 build of a model therefore has a much smaller weight
 * footprint, but an identical {@code RunState} working set compared to any other build.
 * <p>
 * NOTICE: Activation buffers (x, xb, hb, q, k, v, att, logits) are FP32 and memory expensive.
 *         The KV cache can be FP16 or capacity-limited via KvCache/AttentionPattern, but legacy
 *         allocator paths (Llama2, Qwen2, SmolLM3, Gemma2) still use full-size FP32 key/value caches.
 */
public final class RunState {
    public float[] x;      // activation at the current time stamp
    public float[] xb;     // residual branch activation
    public float[] xb2;    // convenience buffer
    public float[] hb;     // hidden dimension in the ffn
    public float[] hb2;    // hidden dimension buffer 2
    public float[] q;      // Query
    public float[] k;      // Key
    public float[] v;      // Value
    public float[] att;    // attention value buffer
    public float[] logits; // output

    // Old FP32 Cache / TODO: Deprecate at some point
    public float[] key_cache;
    public float[] value_cache;

    // New cache; handles larger cache sizes (128k)
    public KvCache kvCache;

    // This is used for RoPE scaling calculations with certain models. Out of context to be stored here, but for convenience’s sake.
    public long promptSize;
}
