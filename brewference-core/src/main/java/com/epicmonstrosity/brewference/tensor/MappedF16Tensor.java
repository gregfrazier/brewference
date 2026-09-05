package com.epicmonstrosity.brewference.tensor;

import java.lang.foreign.MemorySegment;

public final class MappedF16Tensor implements FloatTensor {
    private final MemorySegment data;
    private final long elementCount;

    public MappedF16Tensor(final MemorySegment data, final long elementCount) {
        this.data = TensorMemoryUtils.readonlySlice(data, elementCount, Short.BYTES, "F16");
        this.elementCount = elementCount;
    }

    @Override
    public long elementCount() {
        return elementCount;
    }

    @Override
    public float get(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        return Float.float16ToFloat(data.getAtIndex(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, index));
    }

    public MemorySegment data() {
        return data;
    }
}
