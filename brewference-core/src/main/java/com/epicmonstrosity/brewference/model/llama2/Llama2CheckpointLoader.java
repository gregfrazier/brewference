package com.epicmonstrosity.brewference.model.llama2;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Llama2CheckpointLoader  extends GgufCheckpointLoader {
    public Llama2CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        return this.parseCommonConfig(options);
    }
}