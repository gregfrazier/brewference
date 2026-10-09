package com.epicmonstrosity.brewference.tensor;

public sealed interface Tensor permits QuantizedTensor, FloatTensor, CompositeTensor {
    long elementCount();

    /**
     * Read a single element. For quantized implementations this dequantizes lazily from the
     * block the element lives in; for float implementations it is a plain load.
     */
    float value(long index);

    default float[] toArray() {
        if (elementCount() > Integer.MAX_VALUE) {
            throw new IllegalStateException("Tensor is too large for a float[]: " + elementCount());
        }
        final float[] values = new float[(int) elementCount()];
        for (int i = 0; i < values.length; i++) {
            values[i] = value(i);
        }
        return values;
    }
}
