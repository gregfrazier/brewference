package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Gemma3CheckpointLoader extends GgufCheckpointLoader {
    public Gemma3CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        final Config config = this.parseCommonConfig(options);

        // Hardcode, this is not in the GGUF metadata
        config.setHeadSize(256);

        return config;
    }
}
