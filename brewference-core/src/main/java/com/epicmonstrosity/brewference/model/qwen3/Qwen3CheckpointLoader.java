package com.epicmonstrosity.brewference.model.qwen3;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Qwen3CheckpointLoader extends GgufCheckpointLoader {
    public Qwen3CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        final Config config = this.parseCommonConfig(options);
        config.setHeadSize(128);

        return config;
    }
}