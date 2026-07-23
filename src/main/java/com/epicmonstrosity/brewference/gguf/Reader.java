package com.epicmonstrosity.brewference.gguf;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Reader {
    private static final int MAGIC_NUMBER = 0x46554747; // "GGUF" little-endian
    private final FileChannel channel;
    private final Map<String, Object> metadata = new HashMap<>();
    private final List<TensorInfo> tensors = new ArrayList<>();
    private long dataOffset;

    public Reader(final FileChannel channel) {
        this.channel = channel;
    }

    public void read() throws IOException {
        final ByteBuffer headerBuf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(headerBuf);
        headerBuf.flip();

        final int magic = headerBuf.getInt();
        if (magic != MAGIC_NUMBER) {
            throw new IOException("Not a GGUF file (bad magic number): 0x" + Integer.toHexString(magic));
        }

        final int version = headerBuf.getInt();
        if (version != 2 && version != 3) {
            throw new IOException("Unsupported GGUF version: " + version + ", expected 2 or 3 (Little Endian)");
        }

        final long tensorCount = headerBuf.getLong();
        final long metadataCount = headerBuf.getLong();

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

        dataOffset = (channel.position() + alignment - 1) & -alignment;
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
        tensors.add(new TensorInfo(name, dims, type, offset));
    }

    private Object readValue(final int type) throws IOException {
        switch (type) {
            case 0: { // UINT8
                final ByteBuffer buf = ByteBuffer.allocate(1);
                consumeBuffer(buf);
                buf.flip();
                return buf.get(0) & 0xFF;
            }
            case 1: { // INT8
                final ByteBuffer buf = ByteBuffer.allocate(1);
                consumeBuffer(buf);
                buf.flip();
                return (int) buf.get(0);
            }
            case 2: return readShort() & 0xFFFF; // UINT16
            case 3: return readShort(); // INT16
            case 4: return readInt() & 0xFFFFFFFFL; // UINT32
            case 5: return readInt(); // INT32
            case 6: return readFloat(); // FLOAT32
            case 7: { // BOOL
                final ByteBuffer buf = ByteBuffer.allocate(1);
                consumeBuffer(buf);
                buf.flip();
                return buf.get(0) != 0;
            }
            case 8: return readString(); // STRING
            case 9: { // ARRAY
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
            case 10: return readLong(); // UINT64
            case 11: return readLong(); // INT64
            case 12: return readDouble(); // FLOAT64
            default: throw new IOException("Unknown GGUF metadata type: " + type);
        }
    }

    private String readString() throws IOException {
        final long len = readLong();
        if (len < 0 || len > Integer.MAX_VALUE) {
            throw new IOException("Unreasonable string length: " + len);
        }
        final ByteBuffer buf = ByteBuffer.allocate((int) len);
        consumeBuffer(buf);
        return new String(buf.array(), StandardCharsets.UTF_8);
    }

    private int readInt() throws IOException {
        final ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(buf);
        buf.flip();
        return buf.getInt();
    }

    private long readLong() throws IOException {
        final ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(buf);
        buf.flip();
        return buf.getLong();
    }

    private short readShort() throws IOException {
        final ByteBuffer buf = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(buf);
        buf.flip();
        return buf.getShort();
    }

    private float readFloat() throws IOException {
        final ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(buf);
        buf.flip();
        return buf.getFloat();
    }

    private double readDouble() throws IOException {
        final ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        consumeBuffer(buf);
        buf.flip();
        return buf.getDouble();
    }

    private void consumeBuffer(final ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            final int r = channel.read(buf);
            if (r < 0) {
                throw new EOFException("Unexpected EOF while reading GGUF");
            }
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

        public TensorInfo(final String name, final long[] dims, final int type, final long offset) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.offset = offset;
        }
    }
}
