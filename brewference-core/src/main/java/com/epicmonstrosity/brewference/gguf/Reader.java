package com.epicmonstrosity.brewference.gguf;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Fast, direct MemorySegment-based GGUF header, metadata, and tensor info parser.
 */
public class Reader {
    private static final int MAGIC_NUMBER = 0x46554747; // "GGUF" little-endian

    private static final ValueLayout.OfInt LE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort LE_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble LE_DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final MemorySegment segment;
    private final Map<String, Object> metadata = new HashMap<>();
    private final List<TensorInfo> tensors = new ArrayList<>();
    private long dataOffset;
    private long cursor;

    public Reader(final FileChannel channel) throws IOException {
        Objects.requireNonNull(channel, "channel");
        this.segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), Arena.ofAuto());
    }

    public Reader(final MemorySegment segment) {
        this.segment = Objects.requireNonNull(segment, "segment");
    }

    public void read() throws IOException {
        cursor = 0L;
        if (segment.byteSize() < 24) {
            throw new IOException("Unexpected EOF while reading GGUF header");
        }

        final int magic = readInt();
        if (magic != MAGIC_NUMBER) {
            throw new IOException("Not a GGUF file (bad magic number): 0x" + Integer.toHexString(magic));
        }

        final int version = readInt();
        if (version != 2 && version != 3) {
            throw new IOException("Unsupported GGUF version: " + version + ", expected 2 or 3 (Little Endian)");
        }

        final long tensorCount = readLong();
        final long metadataCount = readLong();

        for (long i = 0; i < metadataCount; i++) {
            readMetadataKV();
        }

        for (long i = 0; i < tensorCount; i++) {
            readTensorInfo();
        }

        final Object alignObj = metadata.getOrDefault("general.alignment", 32L);
        final long alignment = (alignObj instanceof Number) ? ((Number) alignObj).longValue() : 32L;
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IOException("Invalid general.alignment: " + alignment + " (must be power of two)");
        }

        dataOffset = (cursor + alignment - 1) & -alignment;
    }

    private void readMetadataKV() throws IOException {
        final String key = readString();
        final int type = readInt();
        final Object value = readValue(type);
        metadata.put(key, value);
    }

    private void readTensorInfo() throws IOException {
        final String name = readString();
        final int n_dims = readInt();
        if (n_dims < 0 || n_dims > 8) {
            throw new IOException("Unreasonable tensor n_dims=" + n_dims + " for tensor: " + name);
        }

        final long[] dims = new long[n_dims];
        for (int i = 0; i < n_dims; i++) {
            dims[i] = readLong();
        }
        final int type = readInt();
        final long offset = readLong();
        final GgmlType ggmlType = GgmlType.fromId(type);
        tensors.add(new TensorInfo(name, dims, type, offset, ggmlType));
    }

    private Object readValue(final int type) throws IOException {
        return switch (type) {
            case 0 -> readByte() & 0xFF;         // UINT8
            case 1 -> (int) readByte();          // INT8
            case 2 -> readShort() & 0xFFFF;      // UINT16
            case 3 -> (int) readShort();         // INT16
            case 4 -> readInt() & 0xFFFFFFFFL;   // UINT32
            case 5 -> readInt();                 // INT32
            case 6 -> readFloat();               // FLOAT32
            case 7 -> readByte() != 0;           // BOOL
            case 8 -> readString();              // STRING
            case 9 -> readArray();               // ARRAY
            case 10 -> readLong();               // UINT64
            case 11 -> readLong();               // INT64
            case 12 -> readDouble();             // FLOAT64
            default -> throw new IOException("Unknown GGUF metadata type: " + type);
        };
    }

    private List<Object> readArray() throws IOException {
        final int arrayType = readInt();
        final long len = readLong();
        if (len < 0 || len > Integer.MAX_VALUE) {
            throw new IOException("Unreasonable array length: " + len);
        }
        final List<Object> list = new ArrayList<>((int) len);
        for (long i = 0; i < len; i++) {
            list.add(readValue(arrayType));
        }
        return list;
    }

    private byte readByte() throws IOException {
        ensureRemaining(1);
        final byte val = segment.get(ValueLayout.JAVA_BYTE, cursor);
        cursor += 1;
        return val;
    }

    private short readShort() throws IOException {
        ensureRemaining(2);
        final short val = segment.get(LE_SHORT, cursor);
        cursor += 2;
        return val;
    }

    private int readInt() throws IOException {
        ensureRemaining(4);
        final int val = segment.get(LE_INT, cursor);
        cursor += 4;
        return val;
    }

    private long readLong() throws IOException {
        ensureRemaining(8);
        final long val = segment.get(LE_LONG, cursor);
        cursor += 8;
        return val;
    }

    private float readFloat() throws IOException {
        ensureRemaining(4);
        final float val = segment.get(LE_FLOAT, cursor);
        cursor += 4;
        return val;
    }

    private double readDouble() throws IOException {
        ensureRemaining(8);
        final double val = segment.get(LE_DOUBLE, cursor);
        cursor += 8;
        return val;
    }

    private String readString() throws IOException {
        final long len = readLong();
        if (len < 0 || len > Integer.MAX_VALUE) {
            throw new IOException("Unreasonable string length: " + len);
        }
        ensureRemaining(len);
        final byte[] bytes = segment.asSlice(cursor, len).toArray(ValueLayout.JAVA_BYTE);
        cursor += len;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private void ensureRemaining(final long bytes) throws IOException {
        if (cursor + bytes > segment.byteSize()) {
            throw new IOException("Unexpected EOF while reading GGUF");
        }
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public List<TensorInfo> getTensors() {
        return tensors;
    }

    public TensorInfo getTensor(final String name) {
        for (final TensorInfo tensor : tensors) {
            if (tensor.name.equals(name)) {
                return tensor;
            }
        }
        return null;
    }

    public long getDataOffset() {
        return dataOffset;
    }

    public static class TensorInfo {
        public final String name;
        public final long[] dims;
        public final int type;
        public final long offset;
        private final GgmlType ggmlType;

        public TensorInfo(final String name, final long[] dims, final int type, final long offset) {
            this(name, dims, type, offset, GgmlType.fromId(type));
        }

        public TensorInfo(final String name, final long[] dims, final int type, final long offset, final GgmlType ggmlType) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.offset = offset;
            this.ggmlType = ggmlType != null ? ggmlType : GgmlType.fromId(type);
        }

        public GgmlType ggmlType() {
            return ggmlType;
        }

        public GgmlType getGgmlType() {
            return ggmlType;
        }

        @Override
        public String toString() {
            return "TensorInfo{" +
                    "name='" + name + '\'' +
                    ", type=" + type +
                    ", offset=" + offset +
                    ", dims=" + Arrays.toString(dims) +
                    '}';
        }
    }
}
