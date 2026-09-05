package com.epicmonstrosity.brewference.transformer;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.cache.Fp16KvCache;
import com.epicmonstrosity.brewference.transformer.cache.Fp32KvCache;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;

public final class RunStateAllocator {
    private RunStateAllocator() { }

    public static RunState clear(final Config config, final RunState runState) {
        final int dim = config.getTransformerDimensions();
        final int kvDim = (dim * config.getNumKVHeads()) / config.getNumHeads();
        final int maxSeqLen = config.getMaxSequenceLength();
        final int nLayers = config.getNumLayers();

        runState.key_cache = new float[nLayers * maxSeqLen * kvDim];
        runState.value_cache = new float[nLayers * maxSeqLen * kvDim];

        return runState;
    }

    public static RunState clearWithHeadSize(final Config config, final RunState runState) {
        final int headSize = config.getHeadSize();
        final int kvDim = config.getNumKVHeads() * headSize;
        final int maxSeqLen = config.getMaxSequenceLength();
        final int nLayers = config.getNumLayers();

        runState.key_cache = new float[nLayers * maxSeqLen * kvDim];
        runState.value_cache = new float[nLayers * maxSeqLen * kvDim];

        return runState;
    }

    public static RunState allocate(final Config config) {
        final int dim = config.getTransformerDimensions();
        final int kvDim = (dim * config.getNumKVHeads()) / config.getNumHeads();

        return allocate(config, dim, kvDim, dim);
    }

    public static RunState allocateWithHeadSize(final Config config) {
        final int headSize = config.getHeadSize();
        final int qDim = config.getNumHeads() * headSize;
        final int kvDim = config.getNumKVHeads() * headSize;
        final int xbDim = Math.max(config.getTransformerDimensions(), qDim);

        return allocate(config, qDim, kvDim, xbDim);
    }

    public static RunState allocateWithHeadSize(final Config config, final AttentionPattern attentionPattern) {
        final int headSize = config.getHeadSize();
        final int qDim = config.getNumHeads() * headSize;
        final int kvDim = config.getNumKVHeads() * headSize;
        final int xbDim = Math.max(config.getTransformerDimensions(), qDim);

        return allocateWithCache(config, qDim, kvDim, xbDim, attentionPattern, false);
    }

    public static RunState allocateWithFp16Cache(final Config config) {
        final int dim = config.getTransformerDimensions();
        final int kvDim = (dim * config.getNumKVHeads()) / config.getNumHeads();

        return allocateWithFp16Cache(config, dim, kvDim, dim);
    }

    public static RunState allocateWithHeadSizeAndFp16Cache(final Config config) {
        final int headSize = config.getHeadSize();
        final int qDim = config.getNumHeads() * headSize;
        final int kvDim = config.getNumKVHeads() * headSize;
        final int xbDim = Math.max(config.getTransformerDimensions(), qDim);

        return allocateWithFp16Cache(config, qDim, kvDim, xbDim);
    }

    /**
     * @deprecated Use allocateWithFp16Cache or allocateWithFp32Cache instead.
     */
    @Deprecated
    private static RunState allocate(final Config config, final int qDim, final int kvDim, final int xbDim) {
        final RunState runState = new RunState();
        final int dim = config.getTransformerDimensions();
        final int hiddenDim = config.getHiddenDimensions();
        final int vocabSize = config.getVocabSize();
        final int maxSeqLen = config.getMaxSequenceLength();
        final int nLayers = config.getNumLayers();

        runState.x = new float[dim];
        runState.xb = new float[xbDim];
        runState.xb2 = new float[dim];
        runState.hb = new float[hiddenDim];
        runState.hb2 = new float[hiddenDim];
        runState.q = new float[qDim];
        runState.k = new float[kvDim];
        runState.v = new float[kvDim];
        runState.att = new float[config.getNumHeads() * maxSeqLen];
        runState.logits = new float[vocabSize];
        runState.key_cache = new float[nLayers * maxSeqLen * kvDim];
        runState.value_cache = new float[nLayers * maxSeqLen * kvDim];

        return runState;
    }

    private static RunState allocateWithFp16Cache(final Config config, final int qDim, final int kvDim, final int xbDim) {
        return allocateWithCache(config, qDim, kvDim, xbDim, null, true);
    }

    private static RunState allocateWithFp32Cache(final Config config, final int qDim, final int kvDim, final int xbDim) {
        return allocateWithCache(config, qDim, kvDim, xbDim, null, false);
    }

    private static RunState allocateWithCache(final Config config, final int qDim, final int kvDim, final int xbDim,
                                              final AttentionPattern attentionPattern, final boolean fp16) {
        final RunState runState = new RunState();
        final int dim = config.getTransformerDimensions();
        final int hiddenDim = config.getHiddenDimensions();
        final int vocabSize = config.getVocabSize();
        final int maxSeqLen = config.getMaxSequenceLength();
        final int nLayers = config.getNumLayers();

        runState.x = new float[dim];
        runState.xb = new float[xbDim];
        runState.xb2 = new float[dim];
        runState.hb = new float[hiddenDim];
        runState.hb2 = new float[hiddenDim];
        runState.q = new float[qDim];
        runState.k = new float[kvDim];
        runState.v = new float[kvDim];
        runState.att = new float[config.getNumHeads() * maxSeqLen];
        runState.logits = new float[vocabSize];
        final int[] capacity = capacityPerLayer(config, attentionPattern);
        runState.kvCache = fp16 ? new Fp16KvCache(kvDim, capacity) : new Fp32KvCache(kvDim, capacity);

        return runState;
    }

    private static int[] capacityPerLayer(final Config config, final AttentionPattern attentionPattern) {
        final int[] capacity = new int[config.getNumLayers()];
        for (int layer = 0; layer < capacity.length; layer++) {
            capacity[layer] = attentionPattern == null
                    ? config.getMaxSequenceLength()
                    : attentionPattern.cacheCapacityForLayer(layer, config);
            if (capacity[layer] <= 0 || capacity[layer] > config.getMaxSequenceLength()) {
                throw new IllegalArgumentException("Invalid KV-cache capacity for layer " + layer);
            }
        }
        return capacity;
    }
}
