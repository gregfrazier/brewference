package com.epicmonstrosity.brewference.transformer.cache;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.model.gemma3.Gemma3AttentionPattern;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.RunStateAllocator;
import com.epicmonstrosity.brewference.transformer.attention.AttentionEngine;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class Fp32KvCacheTest {
    @Test
    void retainsOnlyTheConfiguredRingWindow() {
        try (final Fp32KvCache cache = new Fp32KvCache(2, new int[]{2})) {
            cache.store(0, 0, new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f});
            cache.store(0, 1, new float[]{5.0f, 6.0f}, new float[]{7.0f, 8.0f});
            cache.store(0, 2, new float[]{9.0f, 10.0f}, new float[]{11.0f, 12.0f});

            assertEquals(1, cache.firstResidentPosition(0, 2));
            assertEquals(29.0f, cache.dotKey(0, 2, new float[]{1.0f, 2.0f}, 0, 0, 2));

            final float[] values = new float[2];
            cache.accumulateValue(0, 2, 0.5f, values, 0, 0, 2);
            assertArrayEquals(new float[]{5.5f, 6.0f}, values);
        }
    }

    @Test
    void allocatesGemmaLocalAndGlobalLayersFromTheAttentionPolicy() {
        final Config config = new Config()
                .setTransformerDimensions(8)
                .setHiddenDimensions(16)
                .setNumLayers(6)
                .setNumHeads(2)
                .setNumKVHeads(1)
                .setHeadSize(4)
                .setMaxSequenceLength(10)
                .setSlidingWindow(3)
                .setVocabSize(32);

        final RunState runState = RunStateAllocator.allocateWithHeadSize(config, new Gemma3AttentionPattern(config));
        try (final Fp32KvCache cache = assertInstanceOf(Fp32KvCache.class, runState.kvCache)) {
            assertEquals(4, cache.firstResidentPosition(0, 6));
            assertEquals(0, cache.firstResidentPosition(5, 6));
        }
    }

    @Test
    void attentionUsesOnlyResidentLocalWindowEntriesAfterWraparound() {
        final Config config = new Config()
                .setTransformerDimensions(1)
                .setHiddenDimensions(2)
                .setNumLayers(1)
                .setNumHeads(1)
                .setNumKVHeads(1)
                .setHeadSize(1)
                .setMaxSequenceLength(5)
                .setSlidingWindow(3)
                .setVocabSize(4);
        final Gemma3AttentionPattern pattern = new Gemma3AttentionPattern(config);
        final RunState runState = RunStateAllocator.allocateWithHeadSize(config, pattern);
        try (final Fp32KvCache ignored = assertInstanceOf(Fp32KvCache.class, runState.kvCache)) {
            for (int position = 0; position <= 3; position++) {
                runState.k[0] = position + 1.0f;
                runState.v[0] = position + 1.0f;
                runState.kvCache.store(0, position, runState.k, runState.v);
            }
            runState.q[0] = 1.0f;

            new AttentionEngine(pattern, config).attend(new LayerContext(3, 0, 0, 0), config, runState, 0);

            final float denominator = (float) (Math.exp(2.0) + Math.exp(3.0) + Math.exp(4.0));
            final float expected = (float) ((2.0 * Math.exp(2.0) + 3.0 * Math.exp(3.0) + 4.0 * Math.exp(4.0))
                    / denominator);
            assertEquals(expected, runState.xb[0], 0.0001f);
        }
    }
}