package com.epicmonstrosity.brewference.model.llama2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

public class Llama2AttentionPattern implements AttentionPattern {
    private final Config config;

    public static int getStartPosition(final LayerContext layerContext, final Config config) {
        return 0;
    }

    public Llama2AttentionPattern(final Config config) {
        this.config = config;
    }

    public int windowFor(final LayerContext layerContext) {
        return getStartPosition(layerContext, config);
    }
}
