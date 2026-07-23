package com.epicmonstrosity.brewference.model.smollm3;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.model.llama2.Llama2AttentionPattern;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.decoder.ByteLevelTokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class SmolLM3ModelRunner extends ModelRunner {

    public SmolLM3ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                new SmolLM3PromptEncoder(),
                new ByteLevelTokenDecoder(new SmolLM3PromptEncoder().buildByteToUnicode()),
                debugConsumer
        );
    }

    @Override
    protected Vocabulary loadVocabulary(final Config config) {
        return VocabLoader.loadVocabWithMerges(config);
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
