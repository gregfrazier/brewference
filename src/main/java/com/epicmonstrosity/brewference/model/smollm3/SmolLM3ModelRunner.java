package com.epicmonstrosity.brewference.model.smollm3;

import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.model.llama2.Llama2AttentionPattern;
import com.epicmonstrosity.brewference.model.smollm.SmolLMTokenDecoder;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;

import java.io.IOException;
import java.util.List;

public class SmolLM3ModelRunner extends ModelRunner {
//    private final Config config;
//    private final QuantizedWeights weights;
//    private final Vocabulary vocab;
//    private final TransformerGraph transformer;
//    private final PromptEncoder encoder;
//    private final TokenDecoder decoder;
//    private final GenerationTokenConsumer debugConsumer;

    public SmolLM3ModelRunner(final GgufCheckpointLoader checkpointLoader,
                             final TokenConsumer debugConsumer) throws IOException {
        super(
                checkpointLoader,
                new SmolLM3PromptEncoder(),
                new SmolLMTokenDecoder(new SmolLM3PromptEncoder().buildByteToUnicode()),
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

//    private String decodeToken(final int token) {
//        final String tokenText = vocab.getIdToToken().get(token);
//        if (decoder != null) {
//            return decoder.decode(tokenText);
//        }
//
//        if (vocab.getSpecialTokens().contains(tokenText))
//            return "\n";
//        return tokenText == null
//                ? "[UNKNOWN]"
//                : tokenText
//                .replace("▁", " ")
//                .replace("<0x0A>", "\n")
//                .replace("<0x0D>", "\r")
//                .replace("<0x09>", "\t");
//    }
//
//    private final class Llama2ModelSession implements ModelSession {
//        private final Sampler sampler = new Sampler(System.nanoTime());
//        private final GenerationTokenConsumer debugConsumer;
//        private RunState runState;
//        private int position;
//        private int pendingToken;
//
//        private Llama2ModelSession(final GenerationTokenConsumer debugConsumer) {
//            this.debugConsumer = debugConsumer;
//            reset();
//        }
//
//        @Override
//        public void reset() {
//            runState = Llama2RunState.allocate(config);
//            position = 0;
//            pendingToken = config.getBosToken();
//        }
//
//        @Override
//        public GenerationResult generate(final String prompt,
//                                         final GenerationOptions options) {
//            final List<Integer> promptTokens =
//                    encoder.gptTokenize(vocab, vocab.getTokenToScore(), prompt);
//            debugConsumer.onDebug("Starting prompt prefill processing...");
//            prefill(promptTokens, debugConsumer);
//
//            int generatedTokenCount = 0;
//            boolean eosEncountered = false;
//            final int remainingContext = config.getMaxSequenceLength() - position;
//            final int generationLimit = options.getMaxNewTokens() <= 0
//                    ? remainingContext
//                    : Math.min(options.getMaxNewTokens(), remainingContext);
//
//            debugConsumer.onDebug("Generating...");
//            final long startTime = System.nanoTime();
//            while (!eosEncountered && generatedTokenCount < generationLimit) {
//                final int generatedToken = sampleNextToken(pendingToken, position, options);
//                pendingToken = generatedToken;
//                position++;
//                generatedTokenCount++;
//
//                debugConsumer.onGeneratedToken(position - 1, generatedToken, decodeToken(generatedToken));
//
//                if (generatedToken == config.getEosToken())
//                    eosEncountered = true;
//            }
//            final long endTime = System.nanoTime();
//            final double elapsedTime = endTime - startTime;
//
//            final GenerationResult generationResult = new GenerationResult(promptTokens.size(), generatedTokenCount, eosEncountered, false, elapsedTime);
//            debugConsumer.onComplete(generationResult);
//            return generationResult;
//        }
//
//        @Override
//        public int getPosition() {
//            return position;
//        }
//
//        private int sampleNextToken(
//                final int token,
//                final int tokenPosition,
//                final GenerationOptions options) {
//            transformer.forward(token, tokenPosition, runState, weights);
//
//            if (options.getTemperature() <= 0.0f) {
//                return Kernels.argMax(runState.logits, config.getVocabSize());
//            }
//
//            return sampler.sample(runState.logits, config.getVocabSize(), options.getTemperature(), options.getTopK(), options.getTopP());
//        }
//
//        private void prefill(final List<Integer> promptTokens,
//                             final GenerationTokenConsumer debugConsumer) {
//            for (final Integer promptToken : promptTokens) {
//                ensureContextAvailable();
//                transformer.forward(promptToken, position, runState, weights);
//
//                debugConsumer.onPrefillToken(position, pendingToken, decodeToken(pendingToken));
//
//                pendingToken = promptToken;
//                position++;
//            }
//        }
//
//        private void ensureContextAvailable() {
//            if (position >= config.getMaxSequenceLength()) {
//                throw new IllegalStateException(
//                        "The session has reached its context limit of "
//                                + config.getMaxSequenceLength() + " tokens.");
//            }
//        }
//
//    }
}
