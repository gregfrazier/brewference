package com.epicmonstrosity.brewference.gguf.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.epicmonstrosity.brewference.gguf.data.Config;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * How {@code <arch>.attention.recurrent_layers} reaches {@link Config}, and what a malformed list must
 * report. The key states per layer whether it is a recurrent (Gated Delta Net) layer, which a
 * {@code full_attention_interval} cannot express when the final layer breaks the pattern.
 */
class GgufConfigParserTest {
    private static final String ARCHITECTURE_KEY = "general.architecture";
    private static final String LAYERS_KEY = "qwen35.attention.recurrent_layers";

    /** The FrogNano layout: 33 layers, full attention at 3,7,...,31 and also at 32. */
    private static List<Boolean> recurrentLayers() {
        final Boolean[] layers = new Boolean[33];
        java.util.Arrays.fill(layers, Boolean.TRUE);
        for (final int layer : new int[]{3, 7, 11, 15, 19, 23, 27, 31, 32}) layers[layer] = Boolean.FALSE;
        return List.of(layers);
    }

    private static Map<String, Object> metadata() {
        final Map<String, Object> metadata = new HashMap<>();
        metadata.put(ARCHITECTURE_KEY, "qwen35");
        metadata.put("qwen35.block_count", 33);
        metadata.put("qwen35.full_attention_interval", 4);
        return metadata;
    }

    @Test
    void aStatedLayerListReachesTheConfig() {
        final Map<String, Object> metadata = metadata();
        metadata.put(LAYERS_KEY, recurrentLayers());

        final Config config = GgufConfigParser.parseCommon(metadata);

        assertEquals(33, config.getNumLayers());
        assertTrue(config.hasExplicitRecurrentLayers());
        assertEquals(33, config.getRecurrentLayers().length);
        assertTrue(config.isFullAttentionLayer(32), "the final layer is full attention");
        assertTrue(config.isFullAttentionLayer(31));
        assertFalse(config.isFullAttentionLayer(30));
        assertFalse(config.isFullAttentionLayer(0));
    }

    @Test
    void theParsedConfigArrayIsACopySoCallersCannotMutateThePattern() {
        final Map<String, Object> metadata = metadata();
        metadata.put(LAYERS_KEY, recurrentLayers());
        final Config config = GgufConfigParser.parseCommon(metadata);

        config.getRecurrentLayers()[32] = true;

        assertFalse(config.getRecurrentLayers()[32], "the stored pattern is untouched");
        assertTrue(config.isFullAttentionLayer(32), "so layer 32 is still full attention");
    }

    @Test
    void aListThatDoesNotMatchBlockCountIsRejected() {
        final Map<String, Object> metadata = metadata();
        metadata.put(LAYERS_KEY, java.util.Arrays.asList(new Boolean[32]));

        final IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> GgufConfigParser.parseCommon(metadata));
        assertTrue(error.getMessage().contains(LAYERS_KEY), error.getMessage());
        assertTrue(error.getMessage().contains("32 entries"), error.getMessage());
        assertTrue(error.getMessage().contains("qwen35.block_count is 33"), error.getMessage());
    }

    @Test
    void aNonBooleanEntryIsRejectedWithItsIndex() {
        final Map<String, Object> metadata = metadata();
        final java.util.List<Object> mixed = new java.util.ArrayList<>(recurrentLayers());
        mixed.set(4, "hybrid");
        mixed.set(4, "hybrid");
        metadata.put(LAYERS_KEY, mixed);

        final IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> GgufConfigParser.parseCommon(metadata));
        assertTrue(error.getMessage().contains("entry 4"), error.getMessage());
        assertTrue(error.getMessage().contains("expected a boolean"), error.getMessage());
    }

    @Test
    void aLayerListWithoutABlockCountCannotBeMapped() {
        final Map<String, Object> metadata = new HashMap<>();
        metadata.put(ARCHITECTURE_KEY, "qwen35");
        metadata.put(LAYERS_KEY, recurrentLayers());

        final IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> GgufConfigParser.parseCommon(metadata));
        assertTrue(error.getMessage().contains(LAYERS_KEY), error.getMessage());
        assertTrue(error.getMessage().contains("qwen35.block_count"), error.getMessage());
    }

    @Test
    void aCheckpointWithoutTheKeyKeepsTheIntervalRule() {
        final Config config = GgufConfigParser.parseCommon(metadata());

        assertNull(config.getRecurrentLayers());
        assertFalse(config.hasExplicitRecurrentLayers());
        assertTrue(config.isHybridAttention());
        assertTrue(config.isFullAttentionLayer(3));
        assertFalse(config.isFullAttentionLayer(32), "the interval rule cannot express this layer");
    }
}
