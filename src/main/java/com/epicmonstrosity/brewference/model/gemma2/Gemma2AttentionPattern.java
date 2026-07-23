package com.epicmonstrosity.brewference.model.gemma2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

public class Gemma2AttentionPattern implements AttentionPattern {
    private final Config config;

    public static boolean isGlobalLayer(final LayerContext layerContext) {
        return ((layerContext.getLayerNum() + 1) % 2 == 0);
    }

    public static int getStartPosition(final LayerContext layerContext, final Config config) {
        return Math.toIntExact(isGlobalLayer(layerContext) ? 0 : Math.max(0, layerContext.getTokenPosition() - config.getSlidingWindow() + 1));
    }

    public Gemma2AttentionPattern(final Config config) {
        this.config = config;
    }

    public int windowFor(final LayerContext layerContext) {
        return getStartPosition(layerContext, config);
    }
}
