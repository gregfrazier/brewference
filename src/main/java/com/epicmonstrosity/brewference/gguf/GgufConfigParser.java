package com.epicmonstrosity.brewference.gguf;

import java.util.Map;
import java.util.function.Consumer;

public final class GgufConfigParser {
    private GgufConfigParser() {}

    public static Config parseCommon(final Map<String, Object> metadata) {
        final Config config = new Config();

        // General
        writeConfig(metadata, "general.architecture", String.class, config::setArchitecture);
        writeConfig(metadata, "general.name", String.class, config::setModelName);
        writeConfig(metadata, "general.size_label", String.class, config::setSizeLabel);

        // Tokenizer
        writeConfig(metadata, "tokenizer.ggml.model", String.class, config::setTokenizerModelType);
        writeConfig(metadata, "tokenizer.ggml.tokens", java.util.List.class, v -> {
            config.setVocabSize(v.size());
        });
        writeConfig(metadata, "tokenizer.ggml.pre", String.class, config::setTokenizerPreTokenizer);

        // Tokenizer: Add Token Booleans
        writeConfig(metadata, "tokenizer.ggml.add_bos_token", Boolean.class, config::setAddBosToken);
        writeConfig(metadata, "tokenizer.ggml.add_eos_token", Boolean.class, config::setAddEosToken);
        writeConfig(metadata, "tokenizer.ggml.add_sep_token", Boolean.class, config::setAddSepToken);
        writeConfig(metadata, "tokenizer.ggml.add_space_prefix", Boolean.class, config::setAddSpacePrefix);

        // Tokenizer: Token Ids
        writeConfigInt(metadata, "tokenizer.ggml.bos_token_id", config::setBosToken);
        writeConfigInt(metadata, "tokenizer.ggml.eos_token_id", config::setEosToken);
        writeConfigInt(metadata, "tokenizer.ggml.unknown_token_id", config::setUnknownToken);
        writeConfigInt(metadata, "tokenizer.ggml.separator_token_id", config::setSepToken);
        writeConfigInt(metadata, "tokenizer.ggml.padding_token_id", config::setPaddingToken);

        // Architecture Specific
        writeConfigInt(metadata, appendArch(config, "embedding_length"), config::setTransformerDimensions);
        writeConfigInt(metadata, appendArch(config, "feed_forward_length"), config::setHiddenDimensions);
        writeConfigInt(metadata, appendArch(config, "block_count"), config::setNumLayers);
        writeConfigInt(metadata, appendArch(config, "context_length"), config::setMaxSequenceLength);
        writeConfigInt(metadata, appendArch(config, "context_length"), config::setContextLength);

        // Architecture Specific: Attention
        writeConfigFloat(metadata, appendArch(config, "attention.layer_norm_rms_epsilon"), config::setLayerNormRMSEpsilon);
        writeConfigInt(metadata, appendArch(config, "attention.head_count"), config::setNumHeads);
        writeConfigInt(metadata, appendArch(config, "attention.head_count_kv"), config::setNumKVHeads);
        writeConfigInt(metadata, appendArch(config, "attention.sliding_window"), config::setSlidingWindow);

        writeConfigFloat(metadata, appendArch(config, "rope.freq_base"), config::setRopeFrequencyBase);
        //writeConfigInt(metadata, appendArch(config, "rope.scaling.original_context_length"), config::setRopeFrequencyBase); // Phi3 4096
        //writeConfigFloat(metadata, appendArch(config, "rope.scaling.attn_factor"), config::setRopeFrequencyBase); // Phi3 1.1902381
        //writeConfigInt(metadata, appendArch(config, "rope.dimension_count"), config::setRopeFrequencyBase); // Phi3 96

        // For reading by Per-Model config parser.
        config.setMetadata(metadata);

        return config;
    }

    private static String appendArch(final Config config, final String metadataName) {
        return String.format("%s.%s", config.getArchitecture(), metadataName);
    }

    private static void writeConfigFloat(final Map<String, Object> metadata,
                                         final String key,
                                         final Consumer<Float> consumer) {
        writeConfig(metadata, key, Number.class, v -> {
            consumer.accept(v.floatValue());
        });
    }

    private static void writeConfigInt(final Map<String, Object> metadata,
                                       final String key,
                                       final Consumer<Integer> consumer) {
        writeConfig(metadata, key, Number.class, v -> {
            consumer.accept(v.intValue());
        });
    }

    public static <T> void writeConfig(final Map<String, Object> metadata,
                                       final String key,
                                       final Class<T> type,
                                       final Consumer<T> consumer) {
        if (metadata.containsKey(key))
            consumer.accept(type.cast(metadata.get(key)));
    }
}
