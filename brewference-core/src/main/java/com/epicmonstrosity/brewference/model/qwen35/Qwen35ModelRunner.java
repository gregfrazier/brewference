package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class Qwen35ModelRunner extends ModelRunner {
    /**
     * Shared so the transformer and every run state agree on which layers hold a KV cache.
     */
    private Qwen35AttentionPattern attentionPattern;

    public Qwen35ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final PromptEncoderRegistry.TokenCodec codec,
                             final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                codec,
                debugConsumer
        );
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        this.attentionPattern = new Qwen35AttentionPattern(config);
        return new Qwen35Transformer(config, attentionPattern);
    }

    @Override
    protected RunState allocateRunState(final Config config) {
        final Qwen35AttentionPattern pattern = attentionPattern != null
                ? attentionPattern
                : new Qwen35AttentionPattern(config);
        return new Qwen35RunState(config, pattern);
    }

    @Override
    protected List<Integer> tokenizePrompt(final String prompt) {
        return encoder.processPrompt(vocab, prompt);
    }

    @Override
    public String id() {
        return "qwen35";
    }
}