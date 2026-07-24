package com.epicmonstrosity.brewference.tokenizer.encoder;

import com.epicmonstrosity.brewference.model.gemma3.GemmaPromptEncoder;
import com.epicmonstrosity.brewference.model.llama2.LlamaPromptEncoder;
import com.epicmonstrosity.brewference.model.qwen2.Qwen2PromptEncoder;
import com.epicmonstrosity.brewference.model.smollm.SmolLMPromptEncoder;
import com.epicmonstrosity.brewference.model.smollm3.SmaugBpePromptEncoder;
import com.epicmonstrosity.brewference.tokenizer.decoder.ByteLevelTokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.decoder.SimpleTokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.decoder.TokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PromptEncoderRegistry {
    private final Map<String, TokenCodec> encoders = new LinkedHashMap<>();

    public PromptEncoderRegistry() {
        register("default-llama",
                new TokenCodec(
                        new GemmaPromptEncoder(),
                        new SimpleTokenDecoder(),
                        VocabLoader::loadVocab
                )
        );
        register("null-llama",
                new TokenCodec(
                        new LlamaPromptEncoder(),
                        new SimpleTokenDecoder(),
                        VocabLoader::loadVocab
                )
        );
        register("smollm-gpt2",
                new TokenCodec(
                        new SmolLMPromptEncoder(),
                        new ByteLevelTokenDecoder(new SmolLMPromptEncoder().buildByteToUnicode()),
                        VocabLoader::loadVocab
                )
        );
        register("smaug-bpe-gpt2",
                new TokenCodec(
                        new SmaugBpePromptEncoder(),
                        new ByteLevelTokenDecoder(new SmaugBpePromptEncoder().buildByteToUnicode()),
                        VocabLoader::loadVocabWithMerges
                )
        );
        register("qwen2-gpt2",
                new TokenCodec(
                        new Qwen2PromptEncoder(),
                        new ByteLevelTokenDecoder(new Qwen2PromptEncoder().buildByteToUnicode()),
                        VocabLoader::loadVocabWithMerges
                )
        );
    }

    public void register(final String id, final TokenCodec encoder) {
        encoders.put(id, encoder);
    }

    public TokenCodec get(final String id) {
        final TokenCodec encoder = encoders.get(id);
        if (encoder == null) {
            throw new IllegalArgumentException("Unknown prompt encoder: " + id);
        }

        return encoder;
    }

    public List<TokenCodec> all() {
        return new ArrayList<>(encoders.values());
    }

    public static class TokenCodec {
        private final PromptEncoder encoder;
        private final TokenDecoder decoder;
        private final VocabLoader.VocabularySupplier vocabSupplier;

        public TokenCodec(final PromptEncoder encoder, final TokenDecoder decoder, final VocabLoader.VocabularySupplier vocabSupplier) {
            this.encoder = encoder;
            this.decoder = decoder;
            this.vocabSupplier = vocabSupplier;
        }

        public PromptEncoder getEncoder() {
            return encoder;
        }

        public TokenDecoder getDecoder() {
            return decoder;
        }

        public VocabLoader.VocabularySupplier getVocabSupplier() {
            return vocabSupplier;
        }
    }
}
