package com.epicmonstrosity.brewference.transformer.attention;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.math.Kernels;

import java.util.Arrays;

public class AttentionEngine {
    private final AttentionPattern attentionPattern;
    private final float sqrtHeadSize;
    private final float inverseScale;
    private final int keyValueDim;
    private final int keyValueMul;

    public AttentionEngine(final AttentionPattern attentionPattern, final Config config) {
        this.attentionPattern = attentionPattern;
        this.sqrtHeadSize = attentionPattern.getAttentionScaling(config); //(float) Math.sqrt(config.getHeadSize());
        this.inverseScale = 1.0f / this.sqrtHeadSize;
        this.keyValueDim = config.getNumKVHeads() * config.getHeadSize();
        this.keyValueMul = config.getNumHeads() / config.getNumKVHeads();
    }

    /**
     * Stores the key and value vectors for the current token position into the cache.
     * This method calculates the appropriate offset in the key and value caches
     * and copies the key and value vectors from the runtime state into those caches.
     *
     * @param layerContext The context of the current layer, including the token position,
     *                     layer-specific details, and cache offsets.
     * @param runState     The runtime state object containing the key and value vectors
     *                     to be stored in the cache, as well as the key and value caches.
     */
    public void storeKeyValueInCache(final LayerContext layerContext, final RunState runState) {
        // Use new KV cache, if available.
        if (runState.kvCache != null) {
            runState.kvCache.store(layerContext.getLayerNum(), layerContext.getTokenPosition(), runState.k, runState.v);
            return;
        }
        // Old KV Cache. TODO: to be removed.
        final int cacheOffset = layerContext.getLayerOffset() + layerContext.getTokenPosition() * this.keyValueDim;
        System.arraycopy(runState.k, 0, runState.key_cache, cacheOffset, this.keyValueDim);
        System.arraycopy(runState.v, 0, runState.value_cache, cacheOffset, this.keyValueDim);
    }

    /**
     * Executes the attention mechanism for a specific head in a transformer-like model.
     * This method computes attention scores, applies softmax normalization, and distributes
     * attention over values for a given layer and head number.
     *
     * @param layerContext The context of the current layer, including token position and layer-specific details.
     * @param config       The configuration object containing model settings, such as head size and maximum sequence length.
     * @param runState     The runtime state object that maintains intermediate computations and memory during execution.
     * @param headNum      The index of the attention head being processed.
     */
    public void attend(final LayerContext layerContext, final Config config, final RunState runState, final int headNum) {
        final int headSize = config.getHeadSize();
        final int queryOffset = headNum * headSize;
        final int attentionOffset = headNum * config.getMaxSequenceLength();
        final int configuredStartPosition = attentionPattern.windowFor(layerContext);
        final int tokenPosition = layerContext.getTokenPosition();
        final int keyValueHeadOffset = keyValueHeadOffset(headNum, headSize);
        final int startPosition = runState.kvCache == null
                ? configuredStartPosition
                : Math.max(configuredStartPosition,
                        runState.kvCache.firstResidentPosition(layerContext.getLayerNum(), tokenPosition));

        if (runState.kvCache != null)
            computeAttentionScoresKvCache(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);
        else
            computeAttentionScores(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);

        Kernels.softMax(runState.att, attentionOffset + startPosition, tokenPosition - startPosition + 1);

        if (runState.kvCache != null)
            applyAttentionToValuesKvCache(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);
        else
            applyAttentionToValues(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);
    }

    @Deprecated
    private void computeAttentionScores(final LayerContext layerContext, final RunState runState,
                                        final int queryOffset, final int attentionOffset,
                                        final int keyValueHeadOffset, final int headSize,
                                        final int startPosition, final int tokenPosition) {
        final float[] q = runState.q;
        final float[] keyCache = runState.key_cache;
        final float[] att = runState.att;

        for (int position = startPosition; position <= tokenPosition; position++) {
            final int keyCacheOffset = layerContext.getLayerOffset() + position * keyValueDim + keyValueHeadOffset;
            float score = 0.0f;

            for (int headIndex = 0; headIndex < headSize; headIndex++) {
                score += q[queryOffset + headIndex] * keyCache[keyCacheOffset + headIndex];
            }

            att[attentionOffset + position] = score * inverseScale;
        }
    }

    private void computeAttentionScoresKvCache(final LayerContext layerContext, final RunState runState,
                                               final int queryOffset, final int attentionOffset,
                                               final int keyValueHeadOffset, final int headSize,
                                               final int startPosition, final int tokenPosition) {
        final int layer = layerContext.getLayerNum();
        for (int position = startPosition; position <= tokenPosition; position++) {
            runState.att[attentionOffset + position] = inverseScale * runState.kvCache
                    .dotKey(layer, position, runState.q, queryOffset, keyValueHeadOffset, headSize);
        }
    }

    @Deprecated
    private void applyAttentionToValues(final LayerContext layerContext, final RunState runState,
                                        final int queryOffset, final int attentionOffset,
                                        final int keyValueHeadOffset, final int headSize,
                                        final int startPosition, final int tokenPosition) {
        final int valueCacheOffset = layerContext.getLayerOffset() + keyValueHeadOffset;
        final float[] att = runState.att;
        final float[] valueCache = runState.value_cache;
        final float[] xb = runState.xb;

        Arrays.fill(xb, queryOffset, queryOffset + headSize, 0.0f);

        for (int position = startPosition; position <= tokenPosition; position++) {
            final float attentionWeight = att[attentionOffset + position];
            final int valueBase = valueCacheOffset + position * keyValueDim;

            for (int headIndex = 0; headIndex < headSize; headIndex++) {
                xb[queryOffset + headIndex] += attentionWeight * valueCache[valueBase + headIndex];
            }
        }
    }

    private void applyAttentionToValuesKvCache(final LayerContext layerContext, final RunState runState,
                                               final int queryOffset, final int attentionOffset,
                                               final int keyValueHeadOffset, final int headSize,
                                               final int startPosition, final int tokenPosition) {
        final int layer = layerContext.getLayerNum();
        Arrays.fill(runState.xb, queryOffset, queryOffset + headSize, 0.0f);

        for (int position = startPosition; position <= tokenPosition; position++) {
            runState.kvCache.accumulateValue(layer, position,
                    runState.att[attentionOffset + position],
                    runState.xb, queryOffset, keyValueHeadOffset, headSize);
        }
    }

    private int keyValueHeadOffset(final int headNum, final int headSize) {
        return (headNum / keyValueMul) * headSize;
    }
}
