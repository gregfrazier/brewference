package com.epicmonstrosity.brewference.model;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GenericCheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma2.Gemma2CheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma2.Gemma2ModelRunner;
import com.epicmonstrosity.brewference.model.gemma3.Gemma3CheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma3.Gemma3ModelRunner;
import com.epicmonstrosity.brewference.model.llama2.Llama2CheckpointLoader;
import com.epicmonstrosity.brewference.model.llama2.Llama2ModelRunner;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2CheckpointLoader;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2ModelRunner;
import com.epicmonstrosity.brewference.model.qwen3.Qwen3CheckpointLoader;
import com.epicmonstrosity.brewference.model.qwen3.Qwen3ModelRunner;
import com.epicmonstrosity.brewference.model.smollm3.SmolLM3ModelRunner;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;

import java.io.IOException;

public class ModelRunnerFactory {
    public static ModelRunner createModelRunner(final String filename, final TokenConsumer debugConsumer) throws IOException {
        final Config config = new GenericCheckpointLoader(filename).getConfig();

        final PromptEncoderRegistry.TokenCodec tokenCodec = new PromptEncoderRegistry().get(config.getCodecId());

        // This is a terrible hack to find the model type. Need to rethink this.
        switch (config.getArchitecture()) {
            case "gemma2":
                return new Gemma2ModelRunner(new Gemma2CheckpointLoader(filename), tokenCodec, debugConsumer);
            case "gemma3":
                return new Gemma3ModelRunner(new Gemma3CheckpointLoader(filename), tokenCodec, debugConsumer);
            case "llama":
            case "llama2":
                return new Llama2ModelRunner(new Llama2CheckpointLoader(filename), tokenCodec, debugConsumer);
            case "smollm3":
                return new SmolLM3ModelRunner(new Llama2CheckpointLoader(filename), tokenCodec, debugConsumer);
            // Not ready
            //case "phi3":
            //    return new Phi3ModelRunner(new Phi3CheckpointLoader(filename), debugConsumer);
            case "qwen3":
                return new Qwen3ModelRunner(new Qwen3CheckpointLoader(filename), tokenCodec, debugConsumer);
            case "qwen2":
                return new Qwen2ModelRunner(new Qwen2CheckpointLoader(filename), tokenCodec, debugConsumer);
            default:
                throw new IllegalArgumentException("Unsupported architecture: " + config.getArchitecture());
        }
    }
}
