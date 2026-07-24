package com.epicmonstrosity.brewference.model.llama2;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.decoder.TokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoder;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class Llama2ModelRunner extends ModelRunner {
    public Llama2ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final PromptEncoderRegistry.TokenCodec tokenCodec,
                             final TokenConsumer debugConsumer) throws IOException {
        super(checkpointLoader, tokenCodec, debugConsumer);
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        return new Llama2Transformer(config, new Llama2AttentionPattern(config));
    }

    @Override
    protected RunState allocateRunState(final Config config) {
        return RunStateAllocator.allocate(config);
    }

    @Override
    protected List<Integer> tokenizePrompt(final String prompt) {
        return encoder.processPrompt(vocab, vocab.getTokenToScore(), prompt);
    }

    @Override
    public String id() {
        return "llama2";
    }
}
