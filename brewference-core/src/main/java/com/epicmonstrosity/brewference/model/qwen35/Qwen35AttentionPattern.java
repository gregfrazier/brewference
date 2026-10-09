package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

/**
 * Attention layout of a hybrid Qwen3.5 model: every {@code full_attention_interval}-th layer is a
 * normal full-attention block, the rest are Gated Delta Net blocks with no KV cache at all.
 * <p>
 * No sliding window — Qwen3.5 full-attention layers attend over the whole context.
 */
public final class Qwen35AttentionPattern implements AttentionPattern {
    private final Config config;

    public Qwen35AttentionPattern(final Config config) {
        this.config = config;
    }

    /**
     * Zero start position: attention begins at token 0 (no local window).
     */
    @Override
    public int windowFor(final LayerContext layerContext) {
        return 0;
    }

    /**
     * GDN layers get capacity {@code 1}, not 0: {@code Fp16KvCache}/{@code Fp32KvCache} reject a
     * non-positive capacity, and a GDN layer never reads or writes its cache, so the single slot is
     * simply never used. This keeps the cache invariants intact instead of teaching every cache
     * implementation about layers that do not have one.
     */
    @Override
    public int cacheCapacityForLayer(final int layer, final Config config) {
        return config.isFullAttentionLayer(layer) ? config.getMaxSequenceLength() : 1;
    }

    /**
     * Stated explicitly (it is also the interface default): Qwen3.5 scales scores by the full
     * 256-wide head, not by the 64 dims RoPE touches.
     */
    @Override
    public float getAttentionScaling(final Config config) {
        return (float) Math.sqrt(config.getHeadSize());
    }

    /**
     * Convenience for the transformer: which branch (full attention vs Gated Delta Net) a layer takes.
     */
    public boolean isFullAttentionLayer(final int layer) {
        return config.isFullAttentionLayer(layer);
    }
}
