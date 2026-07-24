package com.epicmonstrosity.brewference.model.gemma2;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.model.gemma3.GemmaPromptEncoder;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class Gemma2ModelRunner extends ModelRunner {
    public Gemma2ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final PromptEncoderRegistry.TokenCodec tokenCodec,
                             final TokenConsumer debugConsumer) throws IOException {
        super(checkpointLoader, tokenCodec, debugConsumer);
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        return new Gemma2Transformer(config, new Gemma2AttentionPattern(config));
    }

    @Override
    protected RunState allocateRunState(final Config config) {
        return RunStateAllocator.allocateWithHeadSize(config);
    }

    @Override
    protected List<Integer> tokenizePrompt(final String prompt) {
        return encoder.processPrompt(vocab, vocab.getTokenToScore(), prompt);
    }

    @Override
    public String id() {
        return "gemma2";
    }
}
