package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;

import java.io.IOException;
import java.util.Map;

import static com.epicmonstrosity.brewference.gguf.loader.GgufConfigParser.writeConfig;

/*
 * Scans GGUF for the architecture, used to figure out which model runner to use.
 */
public class GenericCheckpointLoader extends GgufCheckpointLoader {

    public GenericCheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        final Map<String, Object> metadata = ggufReader.getMetadata();
        final Config config = new Config();

        writeConfig(metadata, "general.architecture", String.class, config::setArchitecture);
        writeConfig(metadata, "general.name", String.class, config::setModelName);
        writeConfig(metadata, "tokenizer.ggml.model", String.class, config::setTokenizerModelType);
        writeConfig(metadata, "tokenizer.ggml.pre", String.class, config::setTokenizerPreTokenizer);

        return config;
    }
}
