package com.epicmonstrosity.brewference.model.gemma2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Gemma2CheckpointLoader extends GgufCheckpointLoader {
    public Gemma2CheckpointLoader(final String filename) throws IOException {
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
