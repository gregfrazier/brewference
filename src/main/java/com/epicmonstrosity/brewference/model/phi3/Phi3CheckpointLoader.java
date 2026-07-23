package com.epicmonstrosity.brewference.model.phi3;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;

import java.io.IOException;

public class Phi3CheckpointLoader extends GgufCheckpointLoader {
    public Phi3CheckpointLoader(final String filename) throws IOException {
        super(filename);
    }

    @Override
    protected Config parseConfig() {
        return this.parseCommonConfig();
    }
}