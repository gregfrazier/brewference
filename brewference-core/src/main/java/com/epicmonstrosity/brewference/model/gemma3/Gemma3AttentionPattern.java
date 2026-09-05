package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

public class Gemma3AttentionPattern implements AttentionPattern {
    private final Config config;

    public static boolean isGlobalLayer(final LayerContext layerContext) {
        return ((layerContext.getLayerNum() + 1) % 6 == 0);
    }

    public static int getStartPosition(final LayerContext layerContext, final Config config) {
        return Math.toIntExact(isGlobalLayer(layerContext) ? 0 : Math.max(0, layerContext.getTokenPosition() - config.getSlidingWindow() + 1));
    }

    public Gemma3AttentionPattern(final Config config) {
        this.config = config;
    }

    public int windowFor(final LayerContext layerContext) {
        return getStartPosition(layerContext, config);
    }

    @Override
    public float ropePositionScale(final LayerContext layerContext) {
        if (!isGlobalLayer(layerContext) || config.getRopeScalingFactor() <= 0.0f) {
            return 1.0f;
        }
        return 1.0f / config.getRopeScalingFactor();
    }

    @Override
    public float ropeFrequencyBase(final LayerContext layerContext) {
        if (!isGlobalLayer(layerContext)) {
            return 10_000.0f;
        }
        return config.getRopeFrequencyBase() > 0.0f ? config.getRopeFrequencyBase() : 1_000_000.0f;
    }

    @Override
    public int cacheCapacityForLayer(final int layer, final Config config) {
        if ((layer + 1) % 6 == 0 || config.getSlidingWindow() <= 0) {
            return config.getMaxSequenceLength();
        }
        return Math.toIntExact(Math.min(config.getSlidingWindow(), config.getMaxSequenceLength()));
    }
}
