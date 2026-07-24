package com.epicmonstrosity.brewference.model.qwen2;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.decoder.ByteLevelTokenDecoder;
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

public class Qwen2ModelRunner extends ModelRunner {
    public Qwen2ModelRunner(final GgufCheckpointLoader checkpointLoader,
                            final PromptEncoderRegistry.TokenCodec codec,
                            final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                codec,
                //new Qwen2PromptEncoder(),
                //new ByteLevelTokenDecoder(new Qwen2PromptEncoder().buildByteToUnicode()),
                debugConsumer
        );
    }

    @Override
    protected TransformerGraph createTransformer(final Config config) {
        return new Qwen2Transformer(config, new Qwen2AttentionPattern(config));
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
        return "qwen2";
    }
}
