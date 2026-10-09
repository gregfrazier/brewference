package com.epicmonstrosity.brewference.tensor;

/**
 * Concatenation of several {@link Tensor} views read as one logical tensor. Parts are not copied and
 * may mix storage kinds — a mapped F32 norm can sit next to a Q8_0 one, and each part dequantizes
 * lazily when its elements are read.
 */
public final class CompositeTensor implements Tensor {
    private final Tensor[] parts;
    private final long[] offsets;
    private final long elementCount;

    public CompositeTensor(final Tensor... parts) {
        this.parts = TensorMemoryUtils.checkedParts(parts);
        this.offsets = new long[this.parts.length];
        long count = 0;
        for (int i = 0; i < this.parts.length; i++) {
            if (this.parts[i] == null) continue;
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
    public float value(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        final int part = partFor(index);
        return parts[part].value(index - offsets[part]);
    }

    private int partFor(final long index) {
        for (int i = offsets.length - 1; i >= 0; i--) {
            if (index >= offsets[i]) return i;
        }
        throw new AssertionError("Checked composite index has no part");
    }
}
