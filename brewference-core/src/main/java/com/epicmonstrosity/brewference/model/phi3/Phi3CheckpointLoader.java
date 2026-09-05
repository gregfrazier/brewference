package com.epicmonstrosity.brewference.model.phi3;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Phi3CheckpointLoader extends GgufCheckpointLoader {
    public Phi3CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        final Config config = this.parseCommonConfig(options);

        // Can't support more than 8192 context length due to Brewference architecture.
        if (options.getContextLength() > 8192)
            config.setMaxSequenceLength(8192);

        // Not sure what's up with this, but the EOS token is 32000 in the spec, but the model uses 32007.
        config.setEosToken(32007);

        return config;
    }
}