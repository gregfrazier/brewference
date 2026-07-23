package com.epicmonstrosity.brewference.model.llama2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Llama2CheckpointLoader  extends GgufCheckpointLoader {
    public Llama2CheckpointLoader(final String filename) throws IOException {
        super(filename);
    }

    @Override
    protected Config parseConfig() {
        return this.parseCommonConfig();
    }
}