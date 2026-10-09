package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.loader.GgufCheckpointLoader;

import java.io.IOException;

public class Qwen35CheckpointLoader extends GgufCheckpointLoader {
    public Qwen35CheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        super(filename, options);
    }

    @Override
    protected Config parseConfig(final GenerationOptions options) {
        final Config config = this.parseCommonConfig(options);

        // Qwen3.5 heads are key_length wide (256 on the 4B dense model), not the 128 every other
        // Qwen generation uses. Fall back to dim/heads only if the GGUF omitted the key.
        config.setHeadSize(resolveHeadSize(config));

        validateHybridArchitecture(config);

        return config;
    }

    /** Package-private so the head-size rule can be pinned without a GGUF on disk. */
    static int resolveHeadSize(final Config config) {
        final int keyLength = config.getAttentionKeyLength();
        return keyLength > 0 ? keyLength : fallbackHeadSize(config);
    }

    private static int fallbackHeadSize(final Config config) {
        if (config.getNumHeads() <= 0 || config.getTransformerDimensions() <= 0) {
            throw new IllegalArgumentException("%s GGUF is missing attention.head_count / embedding_length and has no attention.key_length to derive headSize from"
                    .formatted(config.getArchitecture()));
        }
        return config.getTransformerDimensions() / config.getNumHeads();
    }

    /**
     * A qwen35 GGUF is only loadable if it describes the SSM/full-attention interleaving; a missing
     * or inconsistent metadata block means the tensor layout cannot be interpreted, so say which
     * number is missing instead of failing later inside a matmul.
     */
    static void validateHybridArchitecture(final Config config) {
        if (config.getFullAttentionInterval() <= 0) {
            throw new IllegalArgumentException("%s GGUF is missing %s.full_attention_interval; cannot determine SSM vs full-attention layers"
                    .formatted(config.getArchitecture(), config.getArchitecture()));
        }
        if (config.getInnerSize() <= 0 || config.getTimeStepRank() <= 0 || config.getGroupCount() <= 0
                || config.getStateSize() <= 0 || config.getConvKernel() <= 0) {
            throw new IllegalArgumentException("%s GGUF is missing SSM metadata: ssm.inner_size=%d ssm.time_step_rank=%d ssm.group_count=%d ssm.state_size=%d ssm.conv_kernel=%d"
                    .formatted(config.getArchitecture(), config.getInnerSize(), config.getTimeStepRank(),
                            config.getGroupCount(), config.getStateSize(), config.getConvKernel()));
        }
        if (config.getTimeStepRank() % config.getGroupCount() != 0) {
            throw new IllegalArgumentException("%s GGUF cannot tile %d group heads across %d time-step heads: time_step_rank must be a multiple of group_count"
                    .formatted(config.getArchitecture(), config.getGroupCount(), config.getTimeStepRank()));
        }
        if (config.getSsmHeadValueDim() != config.getStateSize()) {
            throw new IllegalArgumentException("%s GGUF has inconsistent SSM sizes: inner_size / time_step_rank = %d but state_size = %d"
                    .formatted(config.getArchitecture(), config.getSsmHeadValueDim(), config.getStateSize()));
        }
    }
}