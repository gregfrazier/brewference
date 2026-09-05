package com.epicmonstrosity.brewference.tensor;

import java.util.Objects;

public final class FloatArrayTensor implements FloatTensor {
    private final float[] values;

    public FloatArrayTensor(final float[] values) {
        this.values = Objects.requireNonNull(values, "values");
    }

    @Override
    public long elementCount() {
        return values.length;
    }

    @Override
    public float get(final long index) {
        TensorMemoryUtils.checkIndex(index, values.length);
        return values[(int) index];
    }

    @Override
    public float[] toArray() {
        return values;
    }
}
