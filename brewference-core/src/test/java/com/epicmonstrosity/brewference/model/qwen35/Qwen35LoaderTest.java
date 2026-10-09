package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.data.Config;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Config resolution and the actionable failures a bad qwen35 GGUF should produce.
 */
class Qwen35LoaderTest {

    private static Config baseConfig() {
        return new Config()
                .setArchitecture("qwen35")
                .setTransformerDimensions(2560)
                .setNumHeads(10)
                .setNumKVHeads(2)
                .setNumLayers(32)
                .setFullAttentionInterval(4)
                .setInnerSize(4096)
                .setTimeStepRank(16)
                .setGroupCount(16)
                .setStateSize(256)
                .setConvKernel(4);
    }

    @Test
    void headSizeComesFromKeyLengthWhenPresent() {
        assertEquals(256, Qwen35CheckpointLoader.resolveHeadSize(baseConfig().setAttentionKeyLength(256)));
    }

    @Test
    void headSizeFallsBackToDimensionsDividedByHeadCount() {
        assertEquals(256, Qwen35CheckpointLoader.resolveHeadSize(baseConfig()));
        assertEquals(128, Qwen35CheckpointLoader.resolveHeadSize(baseConfig().setNumHeads(20)));
    }

    @Test
    void missingKeyLengthAndHeadCountIsReported() {
        final Config config = baseConfig().setNumHeads(0).setAttentionKeyLength(0);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35CheckpointLoader.resolveHeadSize(config));
        assertTrue(error.getMessage().contains("attention.head_count"), error.getMessage());
        assertTrue(error.getMessage().contains("attention.key_length"), error.getMessage());
    }

    @Test
    void missingFullAttentionIntervalExplainsWhatCannotBeDecided() {
        final Config config = baseConfig().setFullAttentionInterval(0);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35CheckpointLoader.validateHybridArchitecture(config));
        assertTrue(error.getMessage().contains("full_attention_interval"), error.getMessage());
        assertTrue(error.getMessage().contains("cannot determine SSM vs full-attention layers"), error.getMessage());
    }

    @Test
    void missingSsmMetadataReportsTheActualNumbers() {
        final Config config = baseConfig().setStateSize(0);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35CheckpointLoader.validateHybridArchitecture(config));
        assertTrue(error.getMessage().contains("state_size=0"), error.getMessage());
        assertTrue(error.getMessage().contains("inner_size=4096"), error.getMessage());
    }

    @Test
    void timeStepRankMustBeAMultipleOfGroupCount() {
        final Config config = baseConfig().setGroupCount(6);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35CheckpointLoader.validateHybridArchitecture(config));
        assertTrue(error.getMessage().contains("multiple of group_count"), error.getMessage());
        assertTrue(error.getMessage().contains("6"), error.getMessage());
    }

    @Test
    void innerSizeDividedByTimeStepRankMustMatchStateSize() {
        final Config config = baseConfig().setStateSize(128);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35CheckpointLoader.validateHybridArchitecture(config));
        assertTrue(error.getMessage().contains("state_size = 128"), error.getMessage());
    }

    @Test
    void consistentQwen35MetadataPassesValidation() {
        Qwen35CheckpointLoader.validateHybridArchitecture(baseConfig());
    }

    /** {@code true} = recurrent (Gated Delta Net) layer, as {@code attention.recurrent_layers} states it. */
    private static boolean[] recurrentPattern(final int layers, final int... fullAttentionLayers) {
        final boolean[] pattern = new boolean[layers];
        java.util.Arrays.fill(pattern, true);
        for (final int layer : fullAttentionLayers) pattern[layer] = false;
        return pattern;
    }

    @Test
    void statedRecurrentLayersDecideTheLayerPattern() {
        // A real 33-layer qwen35 layout: interval 4 would say 3,7,...,31, but the checkpoint also
        // makes the final layer a full-attention layer.
        final Config config = baseConfig().setNumLayers(33)
                .setRecurrentLayers(recurrentPattern(33, 3, 7, 11, 15, 19, 23, 27, 31, 32));

        assertTrue(config.hasExplicitRecurrentLayers());
        assertTrue(config.isHybridAttention());
        assertTrue(config.isFullAttentionLayer(32), "the final layer is full attention, not a GDN layer");
        assertTrue(config.isFullAttentionLayer(31));
        assertFalse(config.isFullAttentionLayer(30));
        assertFalse(config.isFullAttentionLayer(0));
        assertEquals(33, config.getRecurrentLayers().length);
    }

    @Test
    void statedListOverridesTheIntervalHeuristic() {
        final Config config = baseConfig().setRecurrentLayers(recurrentPattern(8, 2, 5));

        assertTrue(config.isFullAttentionLayer(2), "the list says so, interval 4 does not");
        assertTrue(config.isFullAttentionLayer(5));
        assertFalse(config.isFullAttentionLayer(3), "interval 4 would have made this full attention");
        assertFalse(config.isFullAttentionLayer(7));
    }

    @Test
    void intervalRuleStillDecidesWhenNoListIsStated() {
        final Config config = baseConfig();

        assertNull(config.getRecurrentLayers());
        assertFalse(config.hasExplicitRecurrentLayers());
        assertTrue(config.isHybridAttention());
        assertTrue(config.isFullAttentionLayer(3));
        assertTrue(config.isFullAttentionLayer(31));
        assertFalse(config.isFullAttentionLayer(32));
        assertFalse(config.isFullAttentionLayer(0));
    }

    @Test
    void aListWithNoRecurrentLayerMeansTheModelIsNotHybrid() {
        final Config config = baseConfig().setNumLayers(4).setRecurrentLayers(recurrentPattern(4, 0, 1, 2, 3));

        assertFalse(config.isHybridAttention(), "nothing is recurrent");
        for (int layer = 0; layer < 4; layer++) assertTrue(config.isFullAttentionLayer(layer));
    }

    @Test
    void askingAboutALayerOutsideTheStatedListIsReported() {
        final Config config = baseConfig().setNumLayers(33)
                .setRecurrentLayers(recurrentPattern(33, 3, 7, 11, 15, 19, 23, 27, 31, 32));

        final IndexOutOfBoundsException error =
                assertThrows(IndexOutOfBoundsException.class, () -> config.isFullAttentionLayer(33));
        assertTrue(error.getMessage().contains("33 entries"), error.getMessage());
        assertTrue(error.getMessage().contains("layer 33"), error.getMessage());
        assertThrows(IndexOutOfBoundsException.class, () -> config.isFullAttentionLayer(-1));
    }
}
