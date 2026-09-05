package com.epicmonstrosity.brewference.tensor;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorPackageTest {

    @Test
    void tensorMemoryUtilsChecksIndex() {
        TensorMemoryUtils.checkIndex(0, 10);
        TensorMemoryUtils.checkIndex(9, 10);
        assertThrows(IndexOutOfBoundsException.class, () -> TensorMemoryUtils.checkIndex(-1, 10));
        assertThrows(IndexOutOfBoundsException.class, () -> TensorMemoryUtils.checkIndex(10, 10));
    }

    @Test
    void tensorMemoryUtilsChecksBlockAligned() {
        TensorMemoryUtils.checkBlockAligned(0, 32, "TEST");
        TensorMemoryUtils.checkBlockAligned(64, 32, "TEST");
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockAligned(-1, 32, "TEST"));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockAligned(33, 32, "TEST"));
    }

    @Test
    void tensorMemoryUtilsChecksBlockSlice() {
        TensorMemoryUtils.checkBlockSlice(0, 32, 64, 32, "TEST");
        TensorMemoryUtils.checkBlockSlice(32, 32, 64, 32, "TEST");
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockSlice(-1, 32, 64, 32, "TEST"));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockSlice(0, 0, 64, 32, "TEST"));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockSlice(1, 32, 64, 32, "TEST"));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockSlice(0, 33, 64, 32, "TEST"));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkBlockSlice(32, 64, 64, 32, "TEST"));
    }

    @Test
    void tensorMemoryUtilsReadonlyBytesAndSlice() {
        final byte[] bytes = new byte[32];
        final MemorySegment writable = MemorySegment.ofArray(bytes);
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.readonlyBytes(writable, 32, "TEST"));

        final MemorySegment readOnly = writable.asReadOnly();
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.readonlyBytes(readOnly, 16, "TEST"));
        final MemorySegment verified = TensorMemoryUtils.readonlyBytes(readOnly, 32, "TEST");
        assertNotNull(verified);

        final MemorySegment sliceVerified = TensorMemoryUtils.readonlySlice(readOnly, 8, Float.BYTES, "F32");
        assertNotNull(sliceVerified);
    }

    @Test
    void tensorMemoryUtilsCheckedParts() {
        assertThrows(NullPointerException.class, () -> TensorMemoryUtils.checkedParts((Object[]) null));
        assertThrows(IllegalArgumentException.class, () -> TensorMemoryUtils.checkedParts(new Object[0]));
        assertThrows(NullPointerException.class, () -> TensorMemoryUtils.checkedParts(new Object[]{null}));

        final String[] parts = new String[]{"a", "b"};
        final String[] copy = TensorMemoryUtils.checkedParts(parts);
        assertArrayEquals(parts, copy);
    }

    @Test
    void floatArrayTensorOperations() {
        final float[] data = new float[]{1.0f, 2.0f, 3.0f};
        final FloatArrayTensor tensor = new FloatArrayTensor(data);
        assertEquals(3, tensor.elementCount());
        assertEquals(1.0f, tensor.get(0));
        assertEquals(3.0f, tensor.get(2));
        assertArrayEquals(data, tensor.toArray());
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.get(3));
        assertThrows(IndexOutOfBoundsException.class, () -> tensor.get(-1));
    }

    @Test
    void compositeFloatTensorOperations() {
        final FloatTensor part1 = new FloatArrayTensor(new float[]{1.0f, 2.0f});
        final FloatTensor part2 = new FloatArrayTensor(new float[]{3.0f, 4.0f, 5.0f});
        final CompositeFloatTensor composite = new CompositeFloatTensor(part1, part2);

        assertEquals(5, composite.elementCount());
        assertEquals(1.0f, composite.get(0));
        assertEquals(2.0f, composite.get(1));
        assertEquals(3.0f, composite.get(2));
        assertEquals(5.0f, composite.get(4));
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, composite.toArray());
        assertThrows(IndexOutOfBoundsException.class, () -> composite.get(5));
        assertThrows(IndexOutOfBoundsException.class, () -> composite.get(-1));
    }

    @Test
    void quantizedSegmentTensorQ8Operations() {
        final byte[] bytes = q8BlockBytes(2.0f, 3, 2);
        final QuantizedSegmentTensor q8 = new QuantizedSegmentTensor(
                MemorySegment.ofArray(bytes).asReadOnly(), GgmlType.Q8_0, 64);
        assertEquals(64, q8.elementCount());
        assertEquals(2, q8.blockCount());
        assertEquals(32, q8.blockSize());
        assertEquals(3, q8.quantizedValue(0));
        assertEquals(2.0f, q8.scale(0));
        assertEquals(6.0f, q8.value(0));
    }

    @Test
    void compositeQuantizedTensorOperations() {
        final QuantizedSegmentTensor q8_1 = new QuantizedSegmentTensor(
                MemorySegment.ofArray(q8BlockBytes(1.0f, 2, 1)).asReadOnly(), GgmlType.Q8_0, 32);
        final QuantizedSegmentTensor q8_2 = new QuantizedSegmentTensor(
                MemorySegment.ofArray(q8BlockBytes(3.0f, 4, 1)).asReadOnly(), GgmlType.Q8_0, 32);

        final CompositeQuantizedTensor composite = new CompositeQuantizedTensor(q8_1, q8_2);
        assertEquals(64, composite.elementCount());
        assertEquals(32, composite.blockSize());
        assertEquals(2, composite.blockCount());
        assertEquals(1.0f, composite.scale(0));
        assertEquals(3.0f, composite.scale(1));
        assertEquals(2.0f, composite.value(0));
        assertEquals(12.0f, composite.value(32));
    }

    private static byte[] q8BlockBytes(final float scale, final int firstValue, final int blocks) {
        final byte[] bytes = new byte[blocks * (Short.BYTES + 32)];
        MemorySegment.ofArray(bytes).set(
                ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                0,
                Float.floatToFloat16(scale));
        bytes[Short.BYTES] = (byte) firstValue;
        return bytes;
    }
}
