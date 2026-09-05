package com.epicmonstrosity.brewference.transformer.cache;

import java.util.Arrays;

public class Fp16KvCache implements KvCache {
    private final short[][] keys;
    private final short[][] values;
    private final int[] capacity;
    private final int kvDim;

    public Fp16KvCache(final int kvDim, final int[] capacityPerLayer) {
        if (kvDim <= 0) {
            throw new IllegalArgumentException("KV dimension must be positive");
        }
        this.kvDim = kvDim;
        this.capacity = capacityPerLayer.clone();
        this.keys = new short[capacity.length][];
        this.values = new short[capacity.length][];
        for (int layer = 0; layer < capacity.length; layer++) {
            if (capacity[layer] <= 0) {
                throw new IllegalArgumentException("Cache capacity must be positive");
            }
            final int size = Math.multiplyExact(capacity[layer], kvDim);
            keys[layer] = new short[size];
            values[layer] = new short[size];
        }
    }

    @Override
    public void store(final int layer, final int position, final float[] k, final float[] v) {
        final int offset = base(layer, position);
        // Slower than FP32 due to non-native array copy
        for (int i = 0; i < kvDim; i++) {
            keys[layer][offset + i] = Float.floatToFloat16(k[i]);
            values[layer][offset + i] = Float.floatToFloat16(v[i]);
        }
    }

    @Override
    public float dotKey(final int layer, final int position, final float[] q,
                        final int qOffset, final int headOffset, final int headSize) {
        final int offset = base(layer, position) + headOffset;
        float score = 0.0f;
        for (int i = 0; i < headSize; i++) {
            score += q[qOffset + i] * Float.float16ToFloat(keys[layer][offset + i]);
        }
        return score;
    }

    @Override
    public void accumulateValue(final int layer, final int position, final float weight,
                                final float[] out, final int outOffset,
                                final int headOffset, final int headSize) {
        final int offset = base(layer, position) + headOffset;
        for (int i = 0; i < headSize; i++) {
            out[outOffset + i] += weight * Float.float16ToFloat(values[layer][offset + i]);
        }
    }

    @Override
    public int firstResidentPosition(final int layer, final int position) {
        return Math.max(0, position - capacity[layer] + 1);
    }

    @Override
    public void close() {
        Arrays.fill(keys, null);
        Arrays.fill(values, null);
    }

    private int base(final int layer, final int position) {
        return Math.multiplyExact(Math.floorMod(position, capacity[layer]), kvDim);
    }
}