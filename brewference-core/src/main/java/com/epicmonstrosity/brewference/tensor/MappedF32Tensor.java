package com.epicmonstrosity.brewference.tensor;

import java.lang.foreign.MemorySegment;

public final class MappedF32Tensor implements FloatTensor {
    private final MemorySegment data;
    private final long elementCount;

    public MappedF32Tensor(final MemorySegment data, final long elementCount) {
        this.data = TensorMemoryUtils.readonlySlice(data, elementCount, Float.BYTES, "F32");
        this.elementCount = elementCount;
    }

    @Override
    public long elementCount() {
        return elementCount;
    }

    @Override
    public float get(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        return data.getAtIndex(TensorMemoryUtils.LITTLE_ENDIAN_FLOAT, index);
    }

    public MemorySegment data() {
        return data;
    }
}
