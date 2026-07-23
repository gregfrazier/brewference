package com.epicmonstrosity.brewference.model.qwen2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Qwen2CheckpointLoader extends GgufCheckpointLoader {
    public Qwen2CheckpointLoader(final String filename) throws IOException {
        super(filename);
    }

    @Override
    protected Config parseConfig() {
        return this.parseCommonConfig();
    }
}