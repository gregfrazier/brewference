package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;
import com.epicmonstrosity.brewference.gguf.loader.GgufConfigParser;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;
import com.epicmonstrosity.brewference.transformer.math.Kernels;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Gemma3NumericsTest {
    @Test
    void appliesRopeScalingOnlyToGlobalAttentionLayers() {
        final Config config = new Config().setRopeFrequencyBase(1_000_000.0f).setRopeScalingFactor(8.0f);
        final Gemma3AttentionPattern pattern = new Gemma3AttentionPattern(config);

        assertEquals(1.0f, pattern.ropePositionScale(new LayerContext(0, 0, 0, 0)));
        assertEquals(0.125f, pattern.ropePositionScale(new LayerContext(0, 5, 0, 0)));
        assertEquals(10_000.0f, pattern.ropeFrequencyBase(new LayerContext(0, 0, 0, 0)));
        assertEquals(1_000_000.0f, pattern.ropeFrequencyBase(new LayerContext(0, 5, 0, 0)));
    }

    @Test
    void readsGemma3RopeScalingFactorFromGgufMetadata() {
        final Config config = GgufConfigParser.parseCommon(Map.of(
                "general.architecture", "gemma3",
                "gemma3.rope.scaling.factor", 8.0f
        ));

        assertEquals(8.0f, config.getRopeScalingFactor());
    }

    @Test
    void gemmaRmsNormAppliesGgufScaleWeightsAsStored() {
        final float[] output = new float[2];
        final float[] input = {3.0f, 4.0f};
        final FloatTensor weights = new FloatArrayTensor(new float[]{1.5f, 0.5f});

        Kernels.rmsNorm(output, input, weights, 0, input.length, 0.0f);

        assertArrayEquals(new float[]{1.2727922f, 0.56568545f}, output, 0.00001f);
    }
}