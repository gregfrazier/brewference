package com.epicmonstrosity.brewference.model.smollm3;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.model.llama2.Llama2AttentionPattern;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class SmolLM3ModelRunner extends ModelRunner {

    public SmolLM3ModelRunner(final GgufCheckpointLoader checkpointLoader,
                              final PromptEncoderRegistry.TokenCodec tokenCodec,
                              final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                tokenCodec,
                debugConsumer
        );
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        return new SmolLM3Transformer(config, new Llama2AttentionPattern(config));
    }

    @Override
    protected RunState allocateRunState(final Config config) {
        return RunStateAllocator.allocate(config);
    }

    @Override
    protected List<Integer> tokenizePrompt(final String prompt) {
        return encoder.processPrompt(vocab, prompt);
    }

    @Override
    public String id() {
        return "smollm3";
    }

}
