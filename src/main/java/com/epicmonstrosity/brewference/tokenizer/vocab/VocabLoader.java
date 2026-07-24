package com.epicmonstrosity.brewference.tokenizer.vocab;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.tui.ChatTui;

import java.util.*;

public class VocabLoader {
    private static final String TOKENIZER_GGML_TOKENS = "tokenizer.ggml.tokens";
    private static final String TOKENIZER_GGML_SCORES = "tokenizer.ggml.scores";
    private static final String TOKENIZER_GGML_MERGES = "tokenizer.ggml.merges";
    private static final String TOKENIZER_GGML_TOKEN_TYPE = "tokenizer.ggml.token_type";

    // The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte).
    private static final Set<Integer> SPECIAL_TOKEN_TYPES = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(2, 3))
    );

    @FunctionalInterface
    public interface VocabularySupplier {
        Vocabulary loadVocab(final Config config);
    }

    @SuppressWarnings("unchecked")
    public static Vocabulary loadVocab(final Config config) {
        if (config.getMetadata().containsKey(TOKENIZER_GGML_MERGES))
            return loadVocabWithMerges(config);

        final Map<String, Object> metadata = config.getMetadata();

        final Vocabulary vocab = new Vocabulary();
        final List<Object> tokens = (List<Object>) metadata.get(TOKENIZER_GGML_TOKENS);
        final List<Float> scores = (List<Float>) metadata.get(TOKENIZER_GGML_SCORES);
        final List<Integer> tokenTypes = (List<Integer>) metadata.get(TOKENIZER_GGML_TOKEN_TYPE);

        // Assume for now that all lists are of equal size
        if (tokens != null && scores != null && tokenTypes != null) {
            for (int i = 0; i < tokens.size(); i++) {
                final String token = (String) tokens.get(i);
                vocab.addTokenId(token, i);
                vocab.addTokenScore(token, scores.get(i));
                if (SPECIAL_TOKEN_TYPES.contains(tokenTypes.get(i))) {
                    vocab.addSpecialTokens(token);
                }
            }
        }

        return vocab;
    }

    @SuppressWarnings("unchecked")
    public static Vocabulary loadVocabWithMerges(final Config config) {
        final Map<String, Object> metadata = config.getMetadata();

        final Vocabulary vocab = new Vocabulary();
        final List<Object> tokens = (List<Object>) metadata.get(TOKENIZER_GGML_TOKENS);
        final List<Integer> tokenTypes = (List<Integer>) metadata.get(TOKENIZER_GGML_TOKEN_TYPE);
        final List<String> merges = (List<String>) metadata.get(TOKENIZER_GGML_MERGES);

        // Assume for now that all lists are of equal size
        if (tokens != null && tokenTypes != null && merges != null) {
            for (int i = 0; i < tokens.size(); i++) {
                final String token = (String) tokens.get(i);
                vocab.addTokenId(token, i);
                if (SPECIAL_TOKEN_TYPES.contains(tokenTypes.get(i))) {
                    vocab.addSpecialTokens(token);
                }
            }
            for (final String merge : merges) {
                vocab.addTokenMerge(merge);
            }
        }
        // Force a build of the merge ranks
        vocab.getMergeRanks();

        return vocab;
    }
}
