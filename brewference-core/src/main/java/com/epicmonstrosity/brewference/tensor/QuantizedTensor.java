package com.epicmonstrosity.brewference.tensor;

public sealed interface QuantizedTensor extends Tensor permits QuantizedSegmentTensor, CompositeQuantizedTensor {
    int blockSize();

    long blockCount();

    float scale(long blockIndex);

    float value(long index);

    QuantizedTensor mappedSlice(long elementOffset, long elements);
}
