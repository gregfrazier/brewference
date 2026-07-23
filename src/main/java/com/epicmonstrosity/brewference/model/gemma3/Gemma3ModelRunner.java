package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class Gemma3ModelRunner extends ModelRunner {
    public Gemma3ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                new GemmaPromptEncoder(),
                null,
                debugConsumer
        );
    }

    @Override
    protected Vocabulary loadVocabulary(final Config config) {
        return VocabLoader.loadVocab(config);
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        return new Gemma3Transformer(config, new Gemma3AttentionPattern(config));
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
        return "gemma3";
    }
}
