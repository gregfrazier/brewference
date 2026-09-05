package com.epicmonstrosity.brewference.tensor;

import com.epicmonstrosity.brewference.gguf.GgmlType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Unified MemorySegment-backed representation for any quantized tensor format.
 */
public final class QuantizedSegmentTensor implements QuantizedTensor {
    private final MemorySegment segment;
    private final GgmlType type;
    private final long elementCount;

    public QuantizedSegmentTensor(final MemorySegment segment, final GgmlType type, final long elementCount) {
        this.type = Objects.requireNonNull(type, "type");
        TensorMemoryUtils.checkBlockAligned(elementCount, type.getBlockSize(), type.name());
        final long expectedBytes = type.byteSizeFor(elementCount);
        this.segment = TensorMemoryUtils.readonlyBytes(segment, expectedBytes, type.name());
        this.elementCount = elementCount;
    }

    @Override
    public long elementCount() {
        return elementCount;
    }

    @Override
    public int blockSize() {
        return type.getBlockSize();
    }

    @Override
    public long blockCount() {
        return elementCount / type.getBlockSize();
    }

    public GgmlType ggmlType() {
        return type;
    }

    public GgmlType type() {
        return type;
    }

    public MemorySegment segment() {
        return segment;
    }

    public MemorySegment data() {
        return segment;
    }

    @Deprecated
    public byte quantizedValue(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        return switch (type) {
            case Q8_0 -> {
                final long block = index / 32;
                yield segment.get(ValueLayout.JAVA_BYTE, block * 34 + Short.BYTES + (index % 32));
            }
            case Q4_0 -> {
                final long block = index / 32;
                final int elem = (int) (index % 32);
                final byte packed = segment.get(ValueLayout.JAVA_BYTE, block * 18 + Short.BYTES + elem % 16);
                final int nibble = elem < 16 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                yield (byte) (nibble - 8);
            }
            default -> throw new UnsupportedOperationException("quantizedValue not supported for: " + type);
        };
    }

    @Override
    public float scale(final long blockIndex) {
        TensorMemoryUtils.checkIndex(blockIndex, blockCount());
        return switch (type) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q1_0 -> Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockIndex * (long) type.getTypeSize()));
            case Q2_K -> Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockIndex * (long) type.getTypeSize() + 80));
            case Q3_K -> Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockIndex * (long) type.getTypeSize() + 108));
            case Q6_K -> Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockIndex * (long) type.getTypeSize() + 208));
            default -> throw new UnsupportedOperationException("Direct scale lookup not supported for: " + type);
        };
    }

    @Override
    public float value(final long index) {
        TensorMemoryUtils.checkIndex(index, elementCount);
        return switch (type) {
            case Q8_0 -> {
                final long block = index / 32;
                final float s = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, block * 34));
                final byte q = segment.get(ValueLayout.JAVA_BYTE, block * 34 + Short.BYTES + (index % 32));
                yield s * q;
            }
            case Q4_0 -> {
                final long block = index / 32;
                final int elem = (int) (index % 32);
                final float s = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, block * 18));
                final byte packed = segment.get(ValueLayout.JAVA_BYTE, block * 18 + Short.BYTES + elem % 16);
                final int nibble = elem < 16 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                yield s * (nibble - 8);
            }
            case Q1_0 -> {
                final long block = index / GgmlType.QK1_0;
                final int elem = (int) (index % GgmlType.QK1_0);
                final float d = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, block * 18));
                final int bits = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, block * 18 + Short.BYTES + elem / 8));
                yield (((bits >>> (elem % 8)) & 1) != 0 ? d : -d);
            }
            case Q4_K -> {
                final long blockIndex = index / 256;
                final int withinBlock = (int) (index % 256);
                final long blockOffset = blockIndex * 144;
                final float d = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset));
                final float dmin = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 2));
                final long scalesOffset = blockOffset + 4;
                final long qsOffset = blockOffset + 16;

                final int group = withinBlock / 64;
                final int inGroup = withinBlock % 64;
                final boolean isHigh = inGroup >= 32;
                final int l = isHigh ? inGroup - 32 : inGroup;
                final int subBlock = isHigh ? group * 2 + 1 : group * 2;

                final int sc = getScaleMinK4(subBlock, segment, scalesOffset, false);
                final int m = getScaleMinK4(subBlock, segment, scalesOffset, true);
                final byte qsByte = segment.get(ValueLayout.JAVA_BYTE, qsOffset + group * 32 + l);
                final int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >>> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
                yield d * sc * quant - dmin * m;
            }
            case Q5_K -> {
                final long blockIndex = index / 256;
                final int withinBlock = (int) (index % 256);
                final long blockOffset = blockIndex * 176;
                final float d = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset));
                final float dmin = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 2));
                final long scalesOffset = blockOffset + 4;
                final long qhOffset = blockOffset + 16;
                final long qsOffset = blockOffset + 48;

                final int group = withinBlock / 64;
                final int inGroup = withinBlock % 64;
                final boolean isHigh = inGroup >= 32;
                final int l = isHigh ? inGroup - 32 : inGroup;
                final int subBlock = isHigh ? group * 2 + 1 : group * 2;

                final int sc = getScaleMinK4(subBlock, segment, scalesOffset, false);
                final int m = getScaleMinK4(subBlock, segment, scalesOffset, true);
                final byte qsByte = segment.get(ValueLayout.JAVA_BYTE, qsOffset + group * 32 + l);
                final int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >>> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
                final int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
                final int qhBit = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qhOffset + l)) >>> qhBitPos) & 1;
                final int quant = nibble | (qhBit << 4);
                yield d * sc * quant - dmin * m;
            }
            case Q2_K -> {
                final long blockIndex = index / 256;
                final int withinBlock = (int) (index % 256);
                final long blockOffset = blockIndex * 84;
                final long scalesOff = blockOffset;
                final long qsOff = blockOffset + 16;
                final float d = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 80));
                final float dmin = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 82));

                final int n = withinBlock / 128;
                final int rem128 = withinBlock % 128;
                final int j = rem128 / 32;
                final int inSub32 = rem128 % 32;
                final boolean isSecondHalf = inSub32 >= 16;
                final int l = isSecondHalf ? inSub32 - 16 : inSub32;
                final int is = n * 8 + j * 2 + (isSecondHalf ? 1 : 0);
                final byte sc = segment.get(ValueLayout.JAVA_BYTE, scalesOff + is);
                final float dl = d * (Byte.toUnsignedInt(sc) & 0xF);
                final float ml = dmin * (Byte.toUnsignedInt(sc) >>> 4);
                final int shift = j * 2;
                final long qByteOff = qsOff + n * 32 + (isSecondHalf ? 16 : 0) + l;
                final int qVal = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qByteOff)) >>> shift) & 3;
                yield dl * qVal - ml;
            }
            case Q3_K -> {
                final long blockIndex = index / 256;
                final int withinBlock = (int) (index % 256);
                final long blockOffset = blockIndex * 110;
                final long hmOff = blockOffset;
                final long qsOff = blockOffset + 32;
                final long scalesOff = blockOffset + 96;
                final float d_all = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 108));

                final int aux0 = segment.get(TensorMemoryUtils.LITTLE_ENDIAN_INT, scalesOff);
                final int aux1 = segment.get(TensorMemoryUtils.LITTLE_ENDIAN_INT, scalesOff + 4);
                final int aux2 = segment.get(TensorMemoryUtils.LITTLE_ENDIAN_INT, scalesOff + 8);
                final int kmask1 = 0x03030303;
                final int kmask2 = 0x0f0f0f0f;
                final int tmp = aux2;
                final int u3 = ((aux1 >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
                final int u2 = ((aux0 >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
                final int u1 = (aux1 & kmask2) | (((tmp >>> 2) & kmask1) << 4);
                final int u0 = (aux0 & kmask2) | (((tmp >>> 0) & kmask1) << 4);

                final int n = withinBlock / 128;
                final int rem128 = withinBlock % 128;
                final int j = rem128 / 32;
                final int inSub32 = rem128 % 32;
                final boolean isSecondHalf = inSub32 >= 16;
                final int l = isSecondHalf ? inSub32 - 16 : inSub32;
                final int is = n * 8 + j * 2 + (isSecondHalf ? 1 : 0);

                final int scaleVal;
                if (is < 4) {
                    scaleVal = (u0 >>> (is * 8)) & 0xFF;
                } else if (is < 8) {
                    scaleVal = (u1 >>> ((is - 4) * 8)) & 0xFF;
                } else if (is < 12) {
                    scaleVal = (u2 >>> ((is - 8) * 8)) & 0xFF;
                } else {
                    scaleVal = (u3 >>> ((is - 12) * 8)) & 0xFF;
                }

                final float dl = d_all * ((scaleVal & 63) - 32);
                final int shift = j * 2;
                final int bitPos = n * 4 + j;
                final int m = 1 << bitPos;
                final long qByteOff = qsOff + n * 32 + (isSecondHalf ? 16 : 0) + l;
                final int qVal = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qByteOff)) >>> shift) & 3;
                final long hmByteOff = hmOff + (isSecondHalf ? 16 : 0) + l;
                final int hmVal = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, hmByteOff));
                final int hVal = (hmVal & m) != 0 ? 0 : 4;
                yield dl * (qVal - hVal);
            }
            case Q6_K -> {
                final long blockIndex = index / 256;
                final int withinBlock = (int) (index % 256);
                final long blockOffset = blockIndex * 210;
                final float d = Float.float16ToFloat(segment.get(TensorMemoryUtils.LITTLE_ENDIAN_SHORT, blockOffset + 208));

                final int half = withinBlock / 128;
                final int rem128 = withinBlock % 128;
                final int sub32 = rem128 / 32;
                final int l = rem128 % 32;

                final long qlBase = blockOffset + half * 64;
                final long qhBase = blockOffset + 128 + half * 32;
                final long scOff = blockOffset + 192;

                final int qlNibble;
                final int qhShift;
                switch (sub32) {
                    case 0 -> {
                        qlNibble = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + l)) & 0xF;
                        qhShift = 0;
                    }
                    case 1 -> {
                        qlNibble = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + 32 + l)) & 0xF;
                        qhShift = 2;
                    }
                    case 2 -> {
                        qlNibble = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + l)) >>> 4) & 0xF;
                        qhShift = 4;
                    }
                    case 3 -> {
                        qlNibble = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + 32 + l)) >>> 4) & 0xF;
                        qhShift = 6;
                    }
                    default -> throw new IllegalStateException();
                }

                final int qhBits = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qhBase + l)) >>> qhShift) & 3;
                final int q6 = (qlNibble | (qhBits << 4)) - 32;
                final int sc = (int) segment.get(ValueLayout.JAVA_BYTE, scOff + half * 8 + sub32 * 2 + l / 16);
                yield d * sc * q6;
            }
            default -> throw new UnsupportedOperationException("Scalar value dequantization not implemented for: " + type);
        };
    }

    public QuantizedSegmentTensor slice(final long elementOffset, final long elements) {
        TensorMemoryUtils.checkBlockSlice(elementOffset, elements, elementCount, type.getBlockSize(), type.name());
        final long byteOffset = type.byteSizeFor(elementOffset);
        final long byteSize = type.byteSizeFor(elements);
        return new QuantizedSegmentTensor(segment.asSlice(byteOffset, byteSize), type, elements);
    }

    @Override
    public QuantizedSegmentTensor mappedSlice(final long elementOffset, final long elements) {
        return slice(elementOffset, elements);
    }

    private static int getScaleMinK4(final int j, final MemorySegment mem, final long scalesOffset, final boolean isMin) {
        if (j < 4) {
            final int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(mem.get(ValueLayout.JAVA_BYTE, scalesOffset + idx)) & 63;
        } else {
            final int lowIdx = j + 4;
            final int highIdx = isMin ? j : j - 4;
            final int low = isMin
                    ? (Byte.toUnsignedInt(mem.get(ValueLayout.JAVA_BYTE, scalesOffset + lowIdx)) >>> 4)
                    : (Byte.toUnsignedInt(mem.get(ValueLayout.JAVA_BYTE, scalesOffset + lowIdx)) & 0xF);
            final int high = (Byte.toUnsignedInt(mem.get(ValueLayout.JAVA_BYTE, scalesOffset + highIdx)) >>> 6) & 0x3;
            return low | (high << 4);
        }
    }
}
