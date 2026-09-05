package com.epicmonstrosity.brewference.transformer.cache;

/**
 * Per-layer key/value ring buffers.
 * <p>
 * A layer's capacity is supplied by its architecture's attention residency policy.
 */
public interface KvCache extends AutoCloseable {
    void store(int layer, int position, float[] k, float[] v);
    float dotKey(int layer, int position, float[] q, int qOffset, int headOffset, int headSize);
    void accumulateValue(int layer, int position, float weight, float[] out, int outOffset, int headOffset, int headSize);
    int firstResidentPosition(int layer, int position);

    @Override
    void close();
}
