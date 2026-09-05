package com.epicmonstrosity.brewference.tensor;

import java.util.Objects;

/**
 * Composite view over multiple QuantizedTensor parts (e.g. multi-layer weights).
 */
public final class CompositeQuantizedTensor implements QuantizedTensor {
    private final QuantizedTensor[] parts;
    private final long[] prefixSums;
    private final long totalElements;
    private final int blockSize;

    public CompositeQuantizedTensor(final QuantizedTensor... parts) {
        Objects.requireNonNull(parts, "parts");
        if (parts.length == 0) {
            throw new IllegalArgumentException("Parts cannot be empty");
        }
        this.parts = parts.clone();
        this.blockSize = parts[0].blockSize();
        this.prefixSums = new long[parts.length + 1];
        long sum = 0;
        for (int i = 0; i < parts.length; i++) {
            Objects.requireNonNull(parts[i], "part " + i);
            if (parts[i].blockSize() != blockSize) {
                throw new IllegalArgumentException("Incompatible block sizes in composite quantized tensor: "
                        + parts[i].blockSize() + " vs " + blockSize);
            }
            prefixSums[i] = sum;
            sum += parts[i].elementCount();
        }
        prefixSums[parts.length] = sum;
        this.totalElements = sum;
    }

    @Override
    public long elementCount() {
        return totalElements;
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public long blockCount() {
        return totalElements / blockSize;
    }

    @Override
    public float scale(final long blockIndex) {
        final long elementOffset = blockIndex * blockSize;
        final int partIndex = findPartIndex(elementOffset);
        final long offsetInPart = elementOffset - prefixSums[partIndex];
        return parts[partIndex].scale(offsetInPart / blockSize);
    }

    @Override
    public float value(final long index) {
        TensorMemoryUtils.checkIndex(index, totalElements);
        final int partIndex = findPartIndex(index);
        final long offsetInPart = index - prefixSums[partIndex];
        return parts[partIndex].value(offsetInPart);
    }

    @Override
    public QuantizedTensor mappedSlice(final long elementOffset, final long elements) {
        TensorMemoryUtils.checkBlockSlice(elementOffset, elements, totalElements, blockSize, "CompositeQuantized");
        final int startPart = findPartIndex(elementOffset);
        final int endPart = findPartIndex(elementOffset + elements - 1);
        if (startPart == endPart) {
            final long offsetInPart = elementOffset - prefixSums[startPart];
            return parts[startPart].mappedSlice(offsetInPart, elements);
        }
        return null;
    }

    public QuantizedTensor[] parts() {
        return parts.clone();
    }

    private int findPartIndex(final long elementIndex) {
        TensorMemoryUtils.checkIndex(elementIndex, totalElements);
        int low = 0;
        int high = parts.length - 1;
        while (low <= high) {
            final int mid = (low + high) >>> 1;
            if (elementIndex < prefixSums[mid]) {
                high = mid - 1;
            } else if (elementIndex >= prefixSums[mid + 1]) {
                low = mid + 1;
            } else {
                return mid;
            }
        }
        throw new IndexOutOfBoundsException("Element index out of bounds: " + elementIndex);
    }
}
