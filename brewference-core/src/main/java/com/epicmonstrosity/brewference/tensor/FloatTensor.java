package com.epicmonstrosity.brewference.tensor;

public sealed interface FloatTensor extends Tensor permits FloatArrayTensor, MappedF32Tensor, MappedF16Tensor, CompositeFloatTensor {
    float get(long index);

    default float[] toArray() {
        if (elementCount() > Integer.MAX_VALUE) {
            throw new IllegalStateException("Tensor is too large for a float[]: " + elementCount());
        }
        final float[] values = new float[(int) elementCount()];
        for (int i = 0; i < values.length; i++) {
            values[i] = get(i);
        }
        return values;
    }
}
