package com.epicmonstrosity.brewference.tensor;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Composite view over multiple QuantizedTensor parts (e.g. one part per transformer layer).
 * <p>
 * Parts may use different GGML formats. Conversions routinely quantize some layers with plain
 * blocks (Q4_0 / Q8_0, 32 elements) and others with K-quants (256 elements), so a single logical
 * weight can legitimately mix block sizes.
 * <p>
 * What stays well defined for a mixed composite:
 * <ul>
 *   <li>{@link #value(long)} always works - it reads through the owning part.</li>
 *   <li>{@link #blockSize()} is the greatest common divisor of the part block sizes, which is the
 *       largest alignment that is legal for every part.</li>
 *   <li>{@link #mappedSlice(long, long)} returns a segment only when the range lies inside one part
 *       and is aligned to that part's own block size; otherwise it returns {@code null} so callers
 *       such as {@code Linear.matmul} can fall back to per-element reads instead of handing a
 *       misaligned range to a block kernel.</li>
 *   <li>{@link #scale(long)} is only meaningful when every part shares one block size and throws
 *       {@link UnsupportedOperationException} otherwise.</li>
 * </ul>
 */
public final class CompositeQuantizedTensor implements QuantizedTensor {
    private final QuantizedTensor[] parts;
    private final long[] prefixSums;
    private final long totalElements;
    private final int blockSize;
    private final boolean uniformBlockSizes;

    public CompositeQuantizedTensor(final QuantizedTensor... parts) {
        Objects.requireNonNull(parts, "parts");
        if (parts.length == 0) {
            throw new IllegalArgumentException("Parts cannot be empty");
        }
        this.parts = parts.clone();
        this.prefixSums = new long[parts.length + 1];
        long sum = 0;
        int alignment = 0;
        int firstBlockSize = -1;
        boolean uniform = true;
        for (int i = 0; i < this.parts.length; i++) {
            Objects.requireNonNull(this.parts[i], "part " + i);
            final int partBlockSize = this.parts[i].blockSize();
            if (partBlockSize <= 0) {
                throw new IllegalArgumentException("Part block size must be positive: " + partBlockSize);
            }
            alignment = alignment == 0 ? partBlockSize : gcd(alignment, partBlockSize);
            if (i == 0) {
                firstBlockSize = partBlockSize;
            } else if (partBlockSize != firstBlockSize) {
                uniform = false;
            }
            prefixSums[i] = sum;
            sum += this.parts[i].elementCount();
        }
        prefixSums[this.parts.length] = sum;
        this.totalElements = sum;
        this.blockSize = alignment;
        this.uniformBlockSizes = uniform;
    }

    @Override
    public long elementCount() {
        return totalElements;
    }

    /**
     * @return the common block size when every part agrees, otherwise the greatest common divisor of
     * the part block sizes, i.e. the strictest alignment that is still legal for every part.
     */
    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public long blockCount() {
        if (uniformBlockSizes) {
            return totalElements / blockSize;
        }
        long blocks = 0;
        for (final QuantizedTensor part : parts) {
            blocks += part.blockCount();
        }
        return blocks;
    }

    @Override
    public float scale(final long blockIndex) {
        if (!uniformBlockSizes) {
            throw new UnsupportedOperationException(
                    "scale(blockIndex) is not defined for a composite with mixed block sizes ("
                            + describeBlockSizes() + "); read element values with value(long) or slice a single part");
        }
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

    /**
     * @return a segment view when the range lies entirely inside one part and is aligned to that
     * part's own block size, otherwise {@code null}.
     * @throws IllegalArgumentException if the range is out of bounds or not aligned to
     * {@link #blockSize()}, which for a mixed composite is the alignment every part can satisfy.
     */
    @Override
    public QuantizedTensor mappedSlice(final long elementOffset, final long elements) {
        TensorMemoryUtils.checkBlockSlice(elementOffset, elements, totalElements, blockSize, "CompositeQuantized");
        final int startPart = findPartIndex(elementOffset);
        final int endPart = findPartIndex(elementOffset + elements - 1);
        if (startPart != endPart) {
            return null;
        }
        final long offsetInPart = elementOffset - prefixSums[startPart];
        final int partBlockSize = parts[startPart].blockSize();
        if (offsetInPart % partBlockSize != 0 || elements % partBlockSize != 0) {
            return null;
        }
        return parts[startPart].mappedSlice(offsetInPart, elements);
    }

    public boolean uniformBlockSizes() {
        return uniformBlockSizes;
    }

    public int partCount() {
        return parts.length;
    }

    public int partBlockSize(final int partIndex) {
        return checkedPart(partIndex).blockSize();
    }

    public long partElementStart(final int partIndex) {
        checkedPart(partIndex);
        return prefixSums[partIndex];
    }

    public long partElementEnd(final int partIndex) {
        checkedPart(partIndex);
        return prefixSums[partIndex + 1];
    }

    /**
     * @return the index of the part that owns {@code elementIndex}.
     */
    public int partIndexFor(final long elementIndex) {
        return findPartIndex(elementIndex);
    }

    public QuantizedTensor[] parts() {
        return parts.clone();
    }

    private QuantizedTensor checkedPart(final int partIndex) {
        if (partIndex < 0 || partIndex >= parts.length) {
            throw new IndexOutOfBoundsException("Part index out of bounds: " + partIndex);
        }
        return parts[partIndex];
    }

    private String describeBlockSizes() {
        final Set<Integer> sizes = new LinkedHashSet<>();
        for (final QuantizedTensor part : parts) {
            sizes.add(part.blockSize());
        }
        return sizes.stream().map(String::valueOf).reduce((a, b) -> a + " vs " + b).orElse("");
    }

    private static int gcd(final int a, final int b) {
        int x = Math.abs(a);
        int y = Math.abs(b);
        while (y != 0) {
            final int t = x % y;
            x = y;
            y = t;
        }
        return x;
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
