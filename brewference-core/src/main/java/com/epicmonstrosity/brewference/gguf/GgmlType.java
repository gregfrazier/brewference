package com.epicmonstrosity.brewference.gguf;

// Pulled from ggml.h
public enum GgmlType {
    F32(0, Float.BYTES, 1),
    F16(1, 2, 1),
    Q4_0(2, 2 + 16 * Byte.BYTES, 32),
    Q8_0(8, 2 + 32 * Byte.BYTES, 32),
    Q2_K(10, 2 * 2 + GgmlType.QK_K / 16 + GgmlType.QK_K / 4, GgmlType.QK_K),
    Q3_K(11, 2 + GgmlType.QK_K / 4 + GgmlType.QK_K / 8 + 12, GgmlType.QK_K),
    Q4_K(12, 2 * 2 + ((GgmlType.QK_K / 16) / 8 * 6) + GgmlType.QK_K / 2, GgmlType.QK_K),
    Q5_K(13, 2 * 2 + ((GgmlType.QK_K / 16) / 8 * 6) + GgmlType.QK_K / 8 + GgmlType.QK_K / 2, GgmlType.QK_K),
    Q6_K(14, GgmlType.QK_K / 2 + GgmlType.QK_K / 4 + GgmlType.QK_K / 16 + 2, GgmlType.QK_K),
    BF16(30, 2, 1),
    Q1_0(41, 2 + GgmlType.QK1_0 / 8, GgmlType.QK1_0),
    Q2_0(42, Integer.MAX_VALUE, 32);

    // Block sizes for supported quantization formats:
    public static final int QK_K = 256;
    public static final int QK1_0 = 128;

    private static final java.util.Map<Integer, GgmlType> BY_ID = new java.util.HashMap<>();

    static {
        for (final GgmlType type : values()) {
            final GgmlType existing = BY_ID.putIfAbsent(type.typeId, type);
            if (existing != null) {
                throw new IllegalStateException("Duplicate GGML type id %d: %s and %s"
                        .formatted(type.typeId, existing.name(), type.name()));
            }
        }
    }

    private final int typeId;
    private final int typeSize;
    private final int blockSize;

    GgmlType(final int typeId, final int typeSize, final int blockSize) {
        if (typeId < 0) {
            throw new IllegalArgumentException("Type id must be non-negative: " + typeId);
        }
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0) {
            throw new IllegalArgumentException("Block size must be a positive power of 2: " + blockSize);
        }
        if (typeSize <= 0) {
            throw new IllegalArgumentException("Type size must be positive: " + typeSize);
        }
        this.typeId = typeId;
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    public int getTypeId() {
        return typeId;
    }

    public int id() {
        return typeId;
    }

    public int getTypeSize() {
        return typeSize;
    }

    public int typeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public int blockSize() {
        return blockSize;
    }

    public boolean isQuantized() {
        return blockSize > 1;
    }

    public static GgmlType fromId(final int id) {
        final GgmlType type = BY_ID.get(id);
        if (type != null) {
            return type;
        }
        throw new UnsupportedOperationException("Unsupported GGML tensor type id: " + id);
    }

    public long byteSizeFor(final long numberOfElements) {
        if (typeSize == Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("Unsupported GGML type size for: " + name());
        }
        final long t = numberOfElements * (long) typeSize;
        if (t % blockSize != 0) {
            throw new IllegalArgumentException("Number of elements %d is not a multiple of block size %d for %s"
                    .formatted(numberOfElements, blockSize, name()));
        }
        return t / blockSize;
    }
}
