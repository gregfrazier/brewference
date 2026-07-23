package com.epicmonstrosity.brewference.transformer.rope;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

import java.util.function.Supplier;

public class NonInterleavedRope {
    public static void apply(final RunState runState, final Config config, final LayerContext layerContext, final Supplier<Float> frequencySupplier) {
        final int headSize = config.getHeadSize();
        final float ropeFrequencyBase = getEffectiveConfig(config.getRopeFrequencyBase(), frequencySupplier.get());
        final float position = calculatePosition(config, layerContext);

        applyToHeads(runState.q, config.getNumHeads(), headSize, headSize, position, ropeFrequencyBase);
        applyToHeads(runState.k, config.getNumKVHeads(), headSize, headSize, position, ropeFrequencyBase);
    }

    private static float getEffectiveConfig(final float config, final Float frequencySupplier) {
        return config > 0.0f ? config : frequencySupplier;
    }

    private static float calculatePosition(final Config config, final LayerContext layerContext) {
        return layerContext.getTokenPosition() * getEffectiveConfig(config.getRopeFrequencyScale(), 1.0f);
    }

    private static void applyToHeads(final float[] vector,
                                     final int headCount,
                                     final int headSize,
                                     final int rotaryDimensionCount,
                                     final float position,
                                     final float frequencyBase) {
        final int half = headSize / 2;
        for (int head = 0; head < headCount; head++) {
            final int headOffset = head * headSize;

            for (int dimension = 0; dimension < half; dimension++) {
                // This could be optimized by precomputing the inverse frequency
                final float inverseFrequency = (float) (1.0 / Math.pow(frequencyBase, (2.0f * dimension) / (float) rotaryDimensionCount));
                final float angle = position * inverseFrequency;
                final float cosine = (float) Math.cos(angle);
                final float sine = (float) Math.sin(angle);

                final int firstIndex = headOffset + dimension;
                final int secondIndex = firstIndex + half;
                final float first = vector[firstIndex];
                final float second = vector[secondIndex];

                vector[firstIndex] = first * cosine - second * sine;
                vector[secondIndex] = first * sine + second * cosine;
            }
        }
    }
}
