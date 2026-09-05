package com.epicmonstrosity.brewference.tensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

public final class TensorMemoryUtils {
    public static final ValueLayout.OfShort LITTLE_ENDIAN_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    public static final ValueLayout.OfInt LITTLE_ENDIAN_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    public static final ValueLayout.OfFloat LITTLE_ENDIAN_FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private TensorMemoryUtils() {}

    public static MemorySegment readonlySlice(final MemorySegment data,
                                              final long elements,
                                              final int bytesPerElement,
                                              final String tensorType) {
        return readonlyBytes(data, Math.multiplyExact(elements, bytesPerElement), tensorType);
    }

    public static MemorySegment readonlyBytes(final MemorySegment data,
                                              final long requiredBytes,
                                              final String tensorType) {
        Objects.requireNonNull(data, "data");
        if (!data.isReadOnly()) {
            throw new IllegalArgumentException("%s tensor memory must be read-only"
                    .formatted(tensorType));
        }
        if (data.byteSize() != requiredBytes) {
            throw new IllegalArgumentException("%s tensor byte size mismatch: expected %d, got %d"
                    .formatted(tensorType, requiredBytes, data.byteSize()));
        }
        return data;
    }

    public static void checkIndex(final long index, final long size) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index %d outside tensor size %d"
                    .formatted(index, size));
        }
    }

    public static void checkBlockAligned(final long elements, final int blockSize, final String tensorType) {
        if (elements < 0 || elements % blockSize != 0) {
            throw new IllegalArgumentException("%s element count must be a non-negative multiple of %d: %d"
                    .formatted(tensorType, blockSize, elements));
        }
    }

    public static void checkBlockSlice(final long elementOffset,
                                       final long elements,
                                       final long size,
                                       final int blockSize,
                                       final String tensorType) {
        if (elementOffset < 0 || elements <= 0 || elementOffset % blockSize != 0 ||
                elements % blockSize != 0 || elementOffset > size - elements) {
            throw new IllegalArgumentException("Invalid %s mapped slice range".formatted(tensorType));
        }
    }

    public static <T> T[] checkedParts(final T[] parts) {
        Objects.requireNonNull(parts, "parts");
        if (parts.length == 0) {
            throw new IllegalArgumentException("Composite tensor requires at least one part");
        }
        final T[] copy = parts.clone();
        for (final T part : copy) {
            Objects.requireNonNull(part, "composite tensor part");
        }
        return copy;
    }
}
