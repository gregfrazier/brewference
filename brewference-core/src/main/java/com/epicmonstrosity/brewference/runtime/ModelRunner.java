package com.epicmonstrosity.brewference.runtime;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.generation.GenerationResult;
import com.epicmonstrosity.brewference.generation.Sampler;
import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.tokenizer.decoder.StreamingUnicodeDecoder;
import com.epicmonstrosity.brewference.tokenizer.decoder.TokenDecoder;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoder;
import com.epicmonstrosity.brewference.tokenizer.encoder.PromptEncoderRegistry;
import com.epicmonstrosity.brewference.tokenizer.vocab.VocabLoader;
import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;
import com.epicmonstrosity.brewference.transformer.math.Kernels;

import java.io.IOException;
import java.util.List;

public abstract class ModelRunner implements AutoCloseable {
    private final GgufCheckpointLoader checkpointLoader;
    protected final Config config;
    protected final QuantizedWeights weights;
    protected final Vocabulary vocab;
    protected final TransformerGraph transformer;
    protected final PromptEncoder encoder;
    protected final TokenDecoder decoder;
    protected final VocabLoader.VocabularySupplier vocabSupplier;
    protected final TokenConsumer debugConsumer;
    private boolean closed;

    protected ModelRunner(final GgufCheckpointLoader checkpointLoader,
                          final PromptEncoderRegistry.TokenCodec codec,
                          final TokenConsumer debugConsumer) throws IOException {
        this.checkpointLoader = checkpointLoader;
        try {
            this.encoder = codec.getEncoder();
            this.decoder = codec.getDecoder();
            this.vocabSupplier = codec.getVocabSupplier();
            this.debugConsumer = debugConsumer;

            debugConsumer.onDebug("Loading weights...");
            checkpointLoader.loadWeights();
            this.config = checkpointLoader.getConfig();
            this.weights = checkpointLoader.getWeights();

            debugConsumer.onDebug("Loading vocabulary...");
            this.vocab = loadVocabulary(config);

            this.transformer = createTransformer(config);
        } catch (final IOException | RuntimeException | Error e) {
            try {
                checkpointLoader.close();
            } catch (final IOException closeException) {
                e.addSuppressed(closeException);
            }
            throw e;
        }
    }

    protected Vocabulary loadVocabulary(final Config config) throws IOException {
        return this.vocabSupplier.loadVocab(config);
    }

    protected abstract TransformerGraph createTransformer(Config config);

    protected abstract RunState allocateRunState(Config config);

    protected abstract List<Integer> tokenizePrompt(String prompt);

    public abstract String id();

    protected String decodeToken(final int token) {
        final String tokenText = vocab.getIdToToken().get(token);
        if (decoder != null) {
            return decoder.decode(tokenText);
        }

        return decodeTokenText(tokenText);
    }

    protected byte[] tokenToBytes(final int token) {
        final String tokenText = vocab.getIdToToken().get(token);
        if (decoder != null) {
            return decoder.tokenToBytes(tokenText);
        }

        return decodeTokenText(tokenText).getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }

    protected String decodeTokenText(final String tokenText) {
        if (vocab.getSpecialTokens().contains(tokenText)) {
            return "\n";
        }

        return tokenText == null
                ? "[UNKNOWN]"
                : tokenText
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
                .replace("<0x0D>", "\r")
                .replace("<0x09>", "\t");
    }

    public ModelSession createSession() {
        ensureOpen();
        return new GenericModelSession();
    }

    @Override
    public void close() throws IOException {
        if (!closed) {
            closed = true;
            checkpointLoader.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Model runner is closed");
        }
    }

    private final class GenericModelSession implements ModelSession {
        private final Sampler sampler = new Sampler();
        private RunState runState;
        private int position;
        private int pendingToken;
        private volatile boolean stopRequested;

        private GenericModelSession() {
            reset();
        }

        @Override
        public Config getConfig() {
            return config;
        }

        @Override
        public void reset() {
            ensureOpen();
            if (runState != null && runState.kvCache != null) {
                runState.kvCache.close();
            }
            runState = allocateRunState(config);
            position = 0;
            pendingToken = config.getBosToken();
            stopRequested = false;
        }

        @Override
        public void requestStop() {
            stopRequested = true;
        }

        @Override
        public GenerationResult generate(final String prompt,
                                         final GenerationOptions options) {
            ensureOpen();
            stopRequested = false;
            final List<Integer> promptTokens = tokenizePrompt(prompt);

            debugConsumer.onDebug("Starting prompt prefill processing...");
            prefill(promptTokens);

            int generatedTokenCount = 0;
            boolean eosEncountered = false;
            final int remainingContext = config.getMaxSequenceLength() - position;
            final int generationLimit = options.getMaxNewTokens() <= 0
                    ? remainingContext
                    : Math.min(options.getMaxNewTokens(), remainingContext);

            debugConsumer.onDebug("Generating...");
            final StreamingUnicodeDecoder streamingDecoder = new StreamingUnicodeDecoder(debugConsumer);

            final long startTime = System.nanoTime();
            while (!eosEncountered && !stopRequested && generatedTokenCount < generationLimit) {
                final int generatedToken = sampleNextToken(pendingToken, position, options);
                pendingToken = generatedToken;
                position++;
                generatedTokenCount++;

                streamingDecoder.append(position - 1, generatedToken, tokenToBytes(generatedToken));

                if (generatedToken == config.getEosToken()) {
                    eosEncountered = true;
                }
            }

            streamingDecoder.flush(position - 1, pendingToken);

            final long endTime = System.nanoTime();
            final double elapsedTime = endTime - startTime;

            final GenerationResult generationResult = new GenerationResult(
                    promptTokens.size(),
                    generatedTokenCount,
                    eosEncountered,
                    false,
                    stopRequested,
                    elapsedTime
            );
            debugConsumer.onComplete(generationResult);
            return generationResult;
        }

        @Override
        public void clear() { }

        @Override
        public int getPosition() {
            return position;
        }

        private int sampleNextToken(final int token,
                                    final int tokenPosition,
                                    final GenerationOptions options) {
            transformer.forward(token, tokenPosition, runState, weights);

            if (options.getTemperature() <= 0.0f) {
                return Kernels.argMax(runState.logits, config.getVocabSize());
            }

            return sampler.sample(
                    runState.logits,
                    config.getVocabSize(),
                    options.getTemperature(),
                    options.getTopK(),
                    options.getTopP()
            );
        }

        private void prefill(final List<Integer> promptTokens) {
            runState.promptSize = promptTokens.size();
            debugConsumer.onPrefillTotal(promptTokens.size());
            for (final Integer promptToken : promptTokens) {
                if (stopRequested) {
                    break;
                }
                ensureContextAvailable();
                transformer.forward(promptToken, position, runState, weights);

                debugConsumer.onPrefillToken(position, pendingToken, decodeToken(pendingToken));

                pendingToken = promptToken;
                position++;
            }
        }

        private void ensureContextAvailable() {
            if (position >= config.getMaxSequenceLength()) {
                throw new IllegalStateException(
                        "The session has reached its context limit of %d tokens."
                                .formatted(config.getMaxSequenceLength())
                );
            }
        }
    }
}
