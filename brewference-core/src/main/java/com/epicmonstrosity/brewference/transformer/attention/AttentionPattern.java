package com.epicmonstrosity.brewference.transformer.attention;

import com.epicmonstrosity.brewference.gguf.data.Config;

public interface AttentionPattern {
    int windowFor(final LayerContext layerContext);

    /**
     * Returns the number of token positions this layer must retain in its KV cache.
     * Architectures with only global attention retain the full model context by default.
     */
    default int cacheCapacityForLayer(final int layer, final Config config) {
        return config.getMaxSequenceLength();
    }

    default float getAttentionScaling(final Config config) {
        return (float) Math.sqrt(config.getHeadSize());
    }

    default float ropePositionScale(final LayerContext layerContext) {
        return 1.0f;
    }
    default float ropeFrequencyBase(final LayerContext layerContext) {
        return 1_000_000.0f;
    }
}
