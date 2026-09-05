package com.epicmonstrosity.brewference.model;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GenericCheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma2.Gemma2CheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma2.Gemma2ModelRunner;
import com.epicmonstrosity.brewference.model.gemma3.Gemma3CheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma3.Gemma3ModelRunner;
import com.epicmonstrosity.brewference.model.llama2.Llama2CheckpointLoader;
import com.epicmonstrosity.brewference.model.llama2.Llama2ModelRunner;
import com.epicmonstrosity.brewference.model.phi3.Phi3CheckpointLoader;
import com.epicmonstrosity.brewference.model.phi3.Phi3ModelRunner;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2CheckpointLoader;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2ModelRunner;
import com.epicmonstrosity.brewference.model.qwen3.Qwen3CheckpointLoader;
import com.epicmonstrosity.brewference.model.qwen3.Qwen3ModelRunner;
import com.epicmonstrosity.brewference.model.smollm3.SmolLM3ModelRunner;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;

import java.io.IOException;

public class ModelRunnerFactory {
    public static ModelRunner createModelRunner(final String filename, final GenerationOptions options, final TokenConsumer debugConsumer) throws IOException {
        final Config config;
        try (final GenericCheckpointLoader checkpointLoader = new GenericCheckpointLoader(filename, options)) {
            config = checkpointLoader.getConfig();
        }

        final PromptEncoderRegistry.TokenCodec tokenCodec = new PromptEncoderRegistry().get(config.getCodecId());

        return switch (config.getArchitecture()) {
            case "gemma2" -> new Gemma2ModelRunner(new Gemma2CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "gemma3" -> new Gemma3ModelRunner(new Gemma3CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "llama", "llama2" ->
                    new Llama2ModelRunner(new Llama2CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "phi3" -> new Phi3ModelRunner(new Phi3CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "qwen2" -> new Qwen2ModelRunner(new Qwen2CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "qwen3" -> new Qwen3ModelRunner(new Qwen3CheckpointLoader(filename, options), tokenCodec, debugConsumer);
            case "smollm3" -> new SmolLM3ModelRunner(new Llama2CheckpointLoader(filename, options), tokenCodec, debugConsumer);

            // TODO: allow unsupported architectures to be interrogated by the configuration viewer
            default -> throw new IllegalArgumentException("Unsupported architecture: " + config.getArchitecture());
        };
    }
}
