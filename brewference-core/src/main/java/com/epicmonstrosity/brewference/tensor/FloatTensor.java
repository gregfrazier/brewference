package com.epicmonstrosity.brewference.tensor;

/**
 * Marker for tensors whose storage is already floating point (F32/F16). The canonical read is
 * {@link Tensor#value(long)}; {@link #get(long)} is kept as an alias so existing call sites on
 * float-typed locals keep compiling.
 */
public sealed interface FloatTensor extends Tensor permits FloatArrayTensor, MappedF32Tensor, MappedF16Tensor {
    default float get(final long index) {
        return value(index);
    }
}
