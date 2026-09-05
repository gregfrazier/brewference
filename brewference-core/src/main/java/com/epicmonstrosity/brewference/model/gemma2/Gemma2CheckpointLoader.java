package com.epicmonstrosity.brewference.model.gemma2;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Gemma2CheckpointLoader extends GgufCheckpointLoader {

    public Gemma2CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
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
