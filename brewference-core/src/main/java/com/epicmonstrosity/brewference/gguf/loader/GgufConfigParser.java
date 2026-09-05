package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.gguf.data.Config;

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
        writeConfig(metadata, "tokenizer.chat_template", String.class, config::setJinjaTemplate);
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
        // ".shortconv.l_cache" // integer defaults to 0
        writeConfigInt(metadata, appendArch(config, "embedding_length_per_layer_input"), config::setEmbeddingLengthPerLayerInput); // default to 0
        writeConfigInt(metadata, appendArch(config, "feed_forward_length"), config::setHiddenDimensions);
        writeConfigInt(metadata, appendArch(config, "block_count"), config::setNumLayers);
        // This is informational only, for most cases we cannot support the full context length due to limitations in the architecture.
        writeConfigInt(metadata, appendArch(config, "context_length"), config::setContextLength);

        // This is writable by the user and must be used for allocation calculations
        writeConfigInt(metadata, appendArch(config, "context_length"), config::setMaxSequenceLength);
        // final_logit_softcapping float

        // Architecture Specific: Attention
        writeConfigFloat(metadata, appendArch(config, "attention.layer_norm_rms_epsilon"), config::setLayerNormRMSEpsilon);
        writeConfigInt(metadata, appendArch(config, "attention.head_count"), config::setNumHeads);
        writeConfigInt(metadata, appendArch(config, "attention.head_count_kv"), config::setNumKVHeads);
        writeConfigInt(metadata, appendArch(config, "attention.sliding_window"), config::setSlidingWindow);

        // RoPE parameters
        writeConfigFloat(metadata, appendArch(config, "rope.freq_base"), config::setRopeFrequencyBase);
        // writeConfigFloat(metadata, appendArch(config, "rope.freq_base_swa"), config::setRopeFrequencyBase);
        writeConfigFloat(metadata, appendArch(config, "rope.scaling.factor"), config::setRopeScalingFactor);
        writeConfigInt(metadata, appendArch(config, "rope.scaling.original_context_length"), config::setRopeScalingOriginalContextLength);
        writeConfigFloat(metadata, appendArch(config, "rope.scaling.attn_factor"), config::setRopeScalingAttnFactor);
        writeConfigInt(metadata, appendArch(config, "rope.dimension_count"), config::setRopeDimCount);

        // MoE parameters (all integer)
        // ".expert_count"
        // ".expert_used_count"
        // ".expert_feed_forward_length"
        // ".leading_dense_block_count" // default to block_count if missing
        // ".expert_gating_func" // 1 = softmax, 2 = sigmoid


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
