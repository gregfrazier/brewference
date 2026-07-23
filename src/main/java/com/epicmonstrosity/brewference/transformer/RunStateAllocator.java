package com.epicmonstrosity.brewference.transformer;

import com.epicmonstrosity.brewference.gguf.Config;

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
}
