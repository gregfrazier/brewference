package com.epicmonstrosity.brewference.model.phi3;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

public class Phi3AttentionPattern implements AttentionPattern {
    private final Config config;

    public static int getStartPosition(final LayerContext layerContext, final Config config) {
        return 0;
    }

    public Phi3AttentionPattern(final Config config) {
        this.config = config;
    }

    public int windowFor(final LayerContext layerContext) {
        return getStartPosition(layerContext, config);
    }

    @Override
    public float getAttentionScaling(final Config config) {
        return (float) (1.0f / Math.sqrt(config.getHeadSize()));
    }
}
