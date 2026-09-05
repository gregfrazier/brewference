package com.epicmonstrosity.brewference.tensor;

public final class CompositeFloatTensor implements FloatTensor {
    private final FloatTensor[] parts;
    private final long[] offsets;
    private final long elementCount;

    public CompositeFloatTensor(final FloatTensor... parts) {
        this.parts = TensorMemoryUtils.checkedParts(parts);
        this.offsets = new long[this.parts.length];
        long count = 0;
        for (int i = 0; i < this.parts.length; i++) {
            offsets[i] = count;
            count = Math.addExact(count, this.parts[i].elementCount());
        }
        this.elementCount = count;
    }

    @Override
    public long elementCount() {
        return elementCount;
    }

    @Override
    public float get(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        final int part = partFor(index);
        return parts[part].get(index - offsets[part]);
    }

    private int partFor(final long index) {
        for (int i = offsets.length - 1; i >= 0; i--) {
            if (index >= offsets[i]) return i;
        }
        throw new AssertionError("Checked composite index has no part");
    }
}
