package com.epicmonstrosity.brewference.transformer.attention;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.math.Kernels;

public class AttentionEngine {
    private final AttentionPattern attentionPattern;
    private final float sqrtHeadSize;
    private final int keyValueDim;
    private final int keyValueMul;

    public AttentionEngine(final AttentionPattern attentionPattern, final Config config) {
        this.attentionPattern = attentionPattern;
        this.sqrtHeadSize = attentionPattern.getAttentionScaling(config); //(float) Math.sqrt(config.getHeadSize());
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
        final int startPosition = attentionPattern.windowFor(layerContext);
        final int tokenPosition = layerContext.getTokenPosition();
        final int keyValueHeadOffset = keyValueHeadOffset(headNum, headSize);

        computeAttentionScores(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);
        Kernels.softMax(runState.att, attentionOffset + startPosition, tokenPosition - startPosition + 1);
        applyAttentionToValues(layerContext, runState, queryOffset, attentionOffset, keyValueHeadOffset, headSize, startPosition, tokenPosition);
    }

    private void computeAttentionScores(final LayerContext layerContext, final RunState runState,
                                        final int queryOffset, final int attentionOffset,
                                        final int keyValueHeadOffset, final int headSize,
                                        final int startPosition, final int tokenPosition) {
        for (int position = startPosition; position <= tokenPosition; position++) {
            final int keyCacheOffset = layerContext.getLayerOffset() + position * keyValueDim + keyValueHeadOffset;
            float score = 0.0f;

            for (int headIndex = 0; headIndex < headSize; headIndex++) {
                score += runState.q[queryOffset + headIndex] * runState.key_cache[keyCacheOffset + headIndex];
            }

            runState.att[attentionOffset + position] = score / sqrtHeadSize;
        }
    }

    private void applyAttentionToValues(final LayerContext layerContext, final RunState runState,
                                        final int queryOffset, final int attentionOffset,
                                        final int keyValueHeadOffset, final int headSize,
                                        final int startPosition, final int tokenPosition) {
        final int valueCacheOffset = layerContext.getLayerOffset() + keyValueHeadOffset;

        for (int headIndex = 0; headIndex < headSize; headIndex++) {
            float weightedValue = 0.0f;

            for (int position = startPosition; position <= tokenPosition; position++) {
                final int valueOffset = valueCacheOffset + position * keyValueDim + headIndex;
                weightedValue += runState.att[attentionOffset + position] * runState.value_cache[valueOffset];
            }

            runState.xb[queryOffset + headIndex] = weightedValue;
        }
    }

    private int keyValueHeadOffset(final int headNum, final int headSize) {
        return (headNum / keyValueMul) * headSize;
    }
}
