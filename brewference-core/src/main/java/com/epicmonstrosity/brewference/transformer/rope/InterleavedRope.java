package com.epicmonstrosity.brewference.transformer.rope;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;

import java.util.function.BiFunction;
import java.util.function.Supplier;

// GPT-J Rope
public class InterleavedRope {
    public static void apply(final RunState runState,
                             final Config config,
                             final LayerContext layerContext,
                             final Supplier<Float> frequencySupplier,
                             final BiFunction<Integer, Integer, RopeCache.Result> resultSupplier) {
        final int headSize = config.getHeadSize();
        final float ropeFrequencyBase = config.getRopeFrequencyBase() > 0.0f
                ? config.getRopeFrequencyBase()
                : frequencySupplier.get();
        final float position = layerContext.getTokenPosition() * (config.getRopeFrequencyScale() > 0.0f
                ? config.getRopeFrequencyScale()
                : 1.0f);

        final RopeCache.Result result = resultSupplier == null
                ? RopeCache.precomputeAngles(headSize, ropeFrequencyBase, position)
                : resultSupplier.apply(headSize, (int) layerContext.getContextLength());

        applyToHeads(runState.q, config.getNumHeads(), headSize, result.cosines, result.sines);
        applyToHeads(runState.k, config.getNumKVHeads(), headSize, result.cosines, result.sines);
    }

    private static void applyToHeads(final float[] vector,
                                     final int headCount,
                                     final int headSize,
                                     final float[] cosines,
                                     final float[] sines) {
        for (int head = 0; head < headCount; head++) {
            final int headOffset = head * headSize;

            for (int dimension = 0; dimension < cosines.length; dimension++) {
                final float cosine = cosines[dimension];
                final float sine = sines[dimension];

                final int firstIndex = headOffset + 2 * dimension;
                final int secondIndex = firstIndex + 1;
                final float first = vector[firstIndex];
                final float second = vector[secondIndex];

                vector[firstIndex] = first * cosine - second * sine;
                vector[secondIndex] = first * sine + second * cosine;
            }
        }
    }
}
