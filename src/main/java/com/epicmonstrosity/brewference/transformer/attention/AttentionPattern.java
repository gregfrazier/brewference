package com.epicmonstrosity.brewference.transformer.attention;

import com.epicmonstrosity.brewference.gguf.Config;

public interface AttentionPattern {
    int windowFor(final LayerContext layerContext);

    default float getAttentionScaling(final Config config) {
        return (float) Math.sqrt(config.getHeadSize());
    }
}
