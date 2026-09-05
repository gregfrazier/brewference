package com.epicmonstrosity.brewference.tensor;

public sealed interface Tensor permits QuantizedTensor, FloatTensor {
    long elementCount();
}
