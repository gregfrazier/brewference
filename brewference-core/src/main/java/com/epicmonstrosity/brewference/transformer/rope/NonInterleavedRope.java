package com.epicmonstrosity.brewference.transformer.rope;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

import java.util.function.BiFunction;
import java.util.function.Supplier;

// NeoX / Split-half Rope
public class NonInterleavedRope {

    public static void apply(final RunState runState,
                             final Config config,
                             final LayerContext layerContext,
                             final Supplier<Float> frequencySupplier,
                             final BiFunction<Integer, Integer, RopeCache.Result> resultSupplier) {
        apply(
                runState,
                config,
                layerContext,
                () -> getEffectiveFloat(config.getRopeFrequencyBase(), frequencySupplier.get()),
                getEffectiveFloat(config.getRopeFrequencyScale(), 1.0f),
                resultSupplier
        );
    }

    public static void apply(final RunState runState,
                             final Config config,
                             final LayerContext layerContext,
                             final Supplier<Float> frequencySupplier,
                             final float positionScale,
                             final BiFunction<Integer, Integer, RopeCache.Result> resultSupplier) {
        final int headSize = config.getHeadSize();
        final float ropeFrequencyBase = frequencySupplier.get();
        final float position = layerContext.getTokenPosition() * positionScale;
        final RopeCache.Result result = resultSupplier == null
                ? RopeCache.precomputeAngles(headSize, ropeFrequencyBase, position)
                : resultSupplier.apply(headSize, (int) layerContext.getContextLength());

        applyToHeads(runState.q, config.getNumHeads(), headSize, result.cosines, result.sines);
        applyToHeads(runState.k, config.getNumKVHeads(), headSize, result.cosines, result.sines);
    }

    private static float getEffectiveFloat(final float config, final Float frequencySupplier) {
        return config > 0.0f ? config : frequencySupplier;
    }


    private static void applyToHeads(final float[] vector,
                                     final int headCount,
                                     final int headSize,
                                     final float[] cosines,
                                     final float[] sines) {
        final int half = cosines.length;
        for (int head = 0; head < headCount; head++) {
            final int headOffset = head * headSize;

            for (int dimension = 0; dimension < half; dimension++) {
                final float cosine = cosines[dimension];
                final float sine = sines[dimension];

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
