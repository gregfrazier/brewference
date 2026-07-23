package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Gemma3CheckpointLoader extends GgufCheckpointLoader {
    public Gemma3CheckpointLoader(final String filename) throws IOException {
        super(filename);
    }

    @Override
    protected Config parseConfig() {
        final Config config = this.parseCommonConfig();

        // Hardcode, this is not in the GGUF metadata
        config.setHeadSize(256);

        return config;
    }
}
