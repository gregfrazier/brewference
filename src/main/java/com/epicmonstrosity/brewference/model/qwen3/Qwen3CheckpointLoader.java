package com.epicmonstrosity.brewference.model.qwen3;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Qwen3CheckpointLoader extends GgufCheckpointLoader {
    public Qwen3CheckpointLoader(final String filename) throws IOException {
        super(filename);
    }

    @Override
    protected Config parseConfig() {
        final Config config = this.parseCommonConfig();
        config.setHeadSize(128);

        return config;
    }
}