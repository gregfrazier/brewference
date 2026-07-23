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
import com.epicmonstrosity.brewference.model.llama2.LlamaPromptEncoder;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2CheckpointLoader;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2ModelRunner;
import com.epicmonstrosity.brewference.model.smollm.SmolLMPromptEncoder;
import com.epicmonstrosity.brewference.model.smollm.SmolLMTokenDecoder;
import com.epicmonstrosity.brewference.model.smollm3.SmolLM3ModelRunner;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoder;

import java.io.IOException;

public class ModelRunnerFactory {
    public static ModelRunner createModelRunner(final String filename, final TokenConsumer debugConsumer) throws IOException {
        final Config config = new GenericCheckpointLoader(filename).getConfig();
        switch (config.getArchitecture()) {
            case "gemma2":
                return new Gemma2ModelRunner(new Gemma2CheckpointLoader(filename), debugConsumer);
            case "gemma3":
                return new Gemma3ModelRunner(new Gemma3CheckpointLoader(filename), debugConsumer);
            case "llama":
            case "llama2": {
                if (config.getTokenizerModelType().equals("gpt2")) { // SmolLM 1/2
                    final PromptEncoder smolLMPromptEncoder = new SmolLMPromptEncoder();
                    return new Llama2ModelRunner(new Llama2CheckpointLoader(filename), smolLMPromptEncoder, new SmolLMTokenDecoder(smolLMPromptEncoder.buildByteToUnicode()), debugConsumer);
                }
                return new Llama2ModelRunner(new Llama2CheckpointLoader(filename), new LlamaPromptEncoder(), null, debugConsumer);
            }
            case "smollm3":
                return new SmolLM3ModelRunner(new Llama2CheckpointLoader(filename), debugConsumer);
            // Not ready
            //case "phi3":
            //    return new Phi3ModelRunner(new Phi3CheckpointLoader(filename), debugConsumer);
            case "qwen2":
                return new Qwen2ModelRunner(new Qwen2CheckpointLoader(filename), debugConsumer);
            default:
                throw new IllegalArgumentException("Unsupported architecture: " + config.getArchitecture());
        }
    }
}
