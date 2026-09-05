package com.epicmonstrosity.brewference.transformer.math;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Modular quantization and floating-point dot-product kernels.
 * TODO: Refactor
 */
public final class QuantizedKernels {

    @FunctionalInterface
    public interface QuantizedDotKernel {
        float dot(MemorySegment weightSegment, long elementOffset, float[] input, int inputOffset, int size);
    }

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;// .SPECIES_256;
    private static final VectorSpecies<Byte> BYTE_SPECIES_256 = ByteVector.SPECIES_256;
    private static final VectorSpecies<Byte> BYTE_SPECIES_128 = ByteVector.SPECIES_128;
    private static final VectorSpecies<Short> SHORT_SPECIES_HALF = ShortVector.SPECIES_128;

    private static final ValueLayout.OfShort LE_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final QuantizedDotKernel[] KERNEL_REGISTRY = new QuantizedDotKernel[GgmlType.values().length];

    /**
     * Sign lookup table for Q1_0: for each of the 256 possible bit patterns, 8 consecutive floats
     * holding {@code +1} where the bit is set and {@code -1} where it is clear. 8 KiB, stays hot in L1.
     * Lets the inner loop be a single {@code fma} per 8 weights instead of per-element scalar math.
     */
    private static final float[] Q1_SIGN_TABLE = buildQ1SignTable();

    private static float[] buildQ1SignTable() {
        final float[] table = new float[256 * 8];
        for (int pattern = 0; pattern < 256; pattern++) {
            for (int bit = 0; bit < 8; bit++) {
                table[(pattern << 3) + bit] = ((pattern >>> bit) & 1) != 0 ? 1.0f : -1.0f;
            }
        }
        return table;
    }

    static {
        registerKernel(GgmlType.Q8_0, QuantizedKernels::dotQ8);
        registerKernel(GgmlType.Q4_0, QuantizedKernels::dotQ4);
        registerKernel(GgmlType.Q6_K, QuantizedKernels::dotQ6K);
        registerKernel(GgmlType.Q4_K, QuantizedKernels::dotQ4K);
        registerKernel(GgmlType.Q5_K, QuantizedKernels::dotQ5K);
        registerKernel(GgmlType.Q2_K, QuantizedKernels::dotQ2K);
        registerKernel(GgmlType.Q3_K, QuantizedKernels::dotQ3K);
        registerKernel(GgmlType.Q1_0, QuantizedKernels::dotQ1);
        registerKernel(GgmlType.F16, QuantizedKernels::dotF16);
        registerKernel(GgmlType.BF16, QuantizedKernels::dotBF16);
        registerKernel(GgmlType.F32, QuantizedKernels::dotF32);
    }

    private QuantizedKernels() {}

    public static void registerKernel(final GgmlType type, final QuantizedDotKernel kernel) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(kernel, "kernel");
        KERNEL_REGISTRY[type.ordinal()] = kernel;
    }

    public static QuantizedDotKernel getKernel(final GgmlType type) {
        final QuantizedDotKernel kernel = KERNEL_REGISTRY[type.ordinal()];
        if (kernel == null) {
            throw new UnsupportedOperationException("No dot-product kernel registered for GGML type: " + type);
        }
        return kernel;
    }

//    public static void matmul(final float[] output, final float[] input, final QuantizedSegmentTensor weights,
//                              final int weightsOffset, final int inputSize, final int outputSize) {
//        final QuantizedDotKernel kernel = getKernel(weights.ggmlType());
//        final MemorySegment segment = weights.segment();
//        final IntStream rows = IntStream.range(0, outputSize);
//        (outputSize < 256 ? rows : rows.parallel()).forEach(row -> {
//            final long rowOffset = (long) weightsOffset + (long) row * inputSize;
//            output[row] = kernel.dot(segment, rowOffset, input, 0, inputSize);
//        });
//    }

    private static final int PARALLEL_THRESHOLD = 256;
    private static final ForkJoinPool MATMUL_POOL =
            new ForkJoinPool(Runtime.getRuntime().availableProcessors());

    public static void matmul(final float[] output, final float[] input, final QuantizedSegmentTensor weights,
                              final int weightsOffset, final int inputSize, final int outputSize) {
        final QuantizedDotKernel kernel = getKernel(weights.ggmlType());
        final MemorySegment segment = weights.segment();

        if (outputSize < PARALLEL_THRESHOLD) {
            for (int row = 0; row < outputSize; row++) {
                output[row] = kernel.dot(segment, (long) weightsOffset + (long) row * inputSize, input, 0, inputSize);
            }
            return;
        }

        final int numWorkers = Math.min(ForkJoinPool.getCommonPoolParallelism(), outputSize);
        final int chunk = (outputSize + numWorkers - 1) / numWorkers;
        final CountDownLatch latch = new CountDownLatch(numWorkers);
        for (int w = 0; w < numWorkers; w++) {
            final int start = w * chunk;
            final int end = Math.min(start + chunk, outputSize);
            if (start >= end) { latch.countDown(); continue; }
            MATMUL_POOL.execute(() -> {
                try {
                    for (int row = start; row < end; row++) {
                        output[row] = kernel.dot(segment, (long) weightsOffset + (long) row * inputSize, input, 0, inputSize);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await(); // wrap InterruptedException per your project's convention
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static float dotQ8(final MemorySegment segment, final long elementOffset,
                              final float[] input, final int inputOffset, final int size) {
        final int blockSize = 32;
        final int typeSize = 34;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        final int vectorLanes = FLOAT_SPECIES.length();
        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final float scale = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final long valueOffset = blockOffset + Short.BYTES;
            final int inBase = inputOffset + b * blockSize;

            FloatVector vecAcc = FloatVector.zero(FLOAT_SPECIES);
            int off = 0;
            for (; off + BYTE_SPECIES_256.length() <= blockSize; off += BYTE_SPECIES_256.length()) {
                final ByteVector qBytes = ByteVector.fromMemorySegment(
                        BYTE_SPECIES_256, segment, valueOffset + off, ByteOrder.LITTLE_ENDIAN);
                for (int part = 0; part < BYTE_SPECIES_256.length() / vectorLanes; part++) {
                    final FloatVector wVec = (FloatVector) qBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, part);
                    final FloatVector inVec = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + off + part * vectorLanes);
                    vecAcc = vecAcc.add(wVec.mul(inVec));
                }
            }
            float blockSum = vecAcc.reduceLanes(VectorOperators.ADD);
            for (; off < blockSize; off++) {
                blockSum += segment.get(ValueLayout.JAVA_BYTE, valueOffset + off) * input[inBase + off];
            }
            total += blockSum * scale;
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float scale = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final long valueOffset = blockOffset + Short.BYTES;
            final int inBase = inputOffset + blocks * blockSize;
            float blockSum = 0.0f;
            for (int i = 0; i < remainder; i++) {
                blockSum += segment.get(ValueLayout.JAVA_BYTE, valueOffset + i) * input[inBase + i];
            }
            total += blockSum * scale;
        }

        return total;
    }

    public static float dotQ4(final MemorySegment segment, final long elementOffset,
                              final float[] input, final int inputOffset, final int size) {
        final int blockSize = 32;
        final int typeSize = 18;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final float scale = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final long valueOffset = blockOffset + Short.BYTES;
            final int inBase = inputOffset + b * blockSize;

            final ByteVector packed = ByteVector.fromMemorySegment(
                    BYTE_SPECIES_128, segment, valueOffset, ByteOrder.LITTLE_ENDIAN);
            final ByteVector loBytes = packed.and((byte) 0x0F).sub((byte) 8);
            final ByteVector hiBytes = packed.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            FloatVector vecAcc = FloatVector.zero(FLOAT_SPECIES);
            final int lanes = FLOAT_SPECIES.length();
            if (lanes == 8) { // 256-bit AVX2
                final FloatVector lo0 = (FloatVector) loBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0);
                final FloatVector lo1 = (FloatVector) loBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, 1);
                final FloatVector hi0 = (FloatVector) hiBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0);
                final FloatVector hi1 = (FloatVector) hiBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, 1);

                final FloatVector in0 = FloatVector.fromArray(FLOAT_SPECIES, input, inBase);
                final FloatVector in1 = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + 8);
                final FloatVector in2 = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + 16);
                final FloatVector in3 = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + 24);

                vecAcc = lo0.mul(in0).add(lo1.mul(in1)).add(hi0.mul(in2)).add(hi1.mul(in3));
            } else {
                for (int part = 0; part < 16 / lanes; part++) {
                    final FloatVector lo = (FloatVector) loBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, part);
                    final FloatVector hi = (FloatVector) hiBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, part);
                    final FloatVector inLo = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + part * lanes);
                    final FloatVector inHi = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + 16 + part * lanes);
                    vecAcc = vecAcc.add(lo.mul(inLo)).add(hi.mul(inHi));
                }
            }
            total += vecAcc.reduceLanes(VectorOperators.ADD) * scale;
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float scale = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final long valueOffset = blockOffset + Short.BYTES;
            final int inBase = inputOffset + blocks * blockSize;
            float blockSum = 0.0f;
            for (int i = 0; i < remainder; i++) {
                final byte packed = segment.get(ValueLayout.JAVA_BYTE, valueOffset + (i % 16));
                final int nibble = i < 16 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                blockSum += (nibble - 8) * input[inBase + i];
            }
            total += blockSum * scale;
        }

        return total;
    }

    public static float dotQ6K(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        final int blockSize = 256;
        final int typeSize = 210;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final long qlOff = blockOffset;
            final long qhOff = blockOffset + 128;
            final long scOff = blockOffset + 192;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 208));

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            for (int h = 0; h < 2; h++) {
                final long qlBase = qlOff + h * 64;
                final long qhBase = qhOff + h * 32;
                final int base = inputOffset + b * blockSize + h * 128;

                for (int c = 0; c < 2; c++) {
                    final ByteVector qlA = ByteVector.fromMemorySegment(
                            BYTE_SPECIES_128, segment, qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    final ByteVector qlB = ByteVector.fromMemorySegment(
                            BYTE_SPECIES_128, segment, qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    final ByteVector qhV = ByteVector.fromMemorySegment(
                            BYTE_SPECIES_128, segment, qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);

                    final ByteVector q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    final ByteVector q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    final ByteVector q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    final ByteVector q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);

                    final float ds0 = d * segment.get(ValueLayout.JAVA_BYTE, scOff + h * 8 + c);
                    final float ds1 = d * segment.get(ValueLayout.JAVA_BYTE, scOff + h * 8 + 2 + c);
                    final float ds2 = d * segment.get(ValueLayout.JAVA_BYTE, scOff + h * 8 + 4 + c);
                    final float ds3 = d * segment.get(ValueLayout.JAVA_BYTE, scOff + h * 8 + 6 + c);

                    final FloatVector ds0Vec = FloatVector.broadcast(FLOAT_SPECIES, ds0);
                    final FloatVector ds1Vec = FloatVector.broadcast(FLOAT_SPECIES, ds1);
                    final FloatVector ds2Vec = FloatVector.broadcast(FLOAT_SPECIES, ds2);
                    final FloatVector ds3Vec = FloatVector.broadcast(FLOAT_SPECIES, ds3);

                    final int sg0Idx = base + c * 16;
                    final int sg1Idx = base + 32 + c * 16;
                    final int sg2Idx = base + 64 + c * 16;
                    final int sg3Idx = base + 96 + c * 16;

                    for (int p = 0; p < 2; p++) {
                        final int off = p * FLOAT_SPECIES.length();
                        final FloatVector q0f = (FloatVector) q0.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector q1f = (FloatVector) q1.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector q2f = (FloatVector) q2.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector q3f = (FloatVector) q3.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);

                        acc = acc.add(q0f.mul(ds0Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, sg0Idx + off)));
                        acc = acc.add(q1f.mul(ds1Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, sg1Idx + off)));
                        acc = acc.add(q2f.mul(ds2Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, sg2Idx + off)));
                        acc = acc.add(q3f.mul(ds3Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, sg3Idx + off)));
                    }
                }
            }
            total += acc.reduceLanes(VectorOperators.ADD);
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 208));
            final long qlBase = blockOffset;
            final long qhBase = blockOffset + 128;
            final long scOff = blockOffset + 192;
            final int inBase = inputOffset + blocks * blockSize;

            for (int i = 0; i < remainder; i++) {
                final int half = i / 128;
                final int rem128 = i % 128;
                final int sub32 = rem128 / 32;
                final int l = rem128 % 32;

                final int qlNibble;
                final int qhShift;
                switch (sub32) {
                    case 0 -> {
                        qlNibble = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + half * 64 + l)) & 0xF;
                        qhShift = 0;
                    }
                    case 1 -> {
                        qlNibble = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + half * 64 + 32 + l)) & 0xF;
                        qhShift = 2;
                    }
                    case 2 -> {
                        qlNibble = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + half * 64 + l)) >>> 4) & 0xF;
                        qhShift = 4;
                    }
                    case 3 -> {
                        qlNibble = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qlBase + half * 64 + 32 + l)) >>> 4) & 0xF;
                        qhShift = 6;
                    }
                    default -> throw new IllegalStateException();
                }

                final int qhBits = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qhBase + half * 32 + l)) >>> qhShift) & 3;
                final int q6 = (qlNibble | (qhBits << 4)) - 32;
                final int sc = (int) segment.get(ValueLayout.JAVA_BYTE, scOff + half * 8 + sub32 * 2 + l / 16);
                total += d * sc * q6 * input[inBase + i];
            }
        }

        return total;
    }

    public static float dotQ4K(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        final int blockSize = 256;
        final int typeSize = 144;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 2));
            final long scalesOff = blockOffset + 4;
            final long qsOff = blockOffset + 16;
            final int inBase = inputOffset + b * blockSize;

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

            for (int g = 0; g < 4; g++) {
                final float d1 = d * getScaleMinK4(g * 2, segment, scalesOff, false);
                final float negM1 = -(dmin * getScaleMinK4(g * 2, segment, scalesOff, true));
                final float d2 = d * getScaleMinK4(g * 2 + 1, segment, scalesOff, false);
                final float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, segment, scalesOff, true));

                final FloatVector d1Vec = FloatVector.broadcast(FLOAT_SPECIES, d1);
                final FloatVector negM1Vec = FloatVector.broadcast(FLOAT_SPECIES, negM1);
                final FloatVector d2Vec = FloatVector.broadcast(FLOAT_SPECIES, d2);
                final FloatVector negM2Vec = FloatVector.broadcast(FLOAT_SPECIES, negM2);

                final int loBase = inBase + g * 64;
                final int hiBase = inBase + g * 64 + 32;
                final long groupQsOff = qsOff + (long) g * 32;

                for (int c = 0; c < 2; c++) {
                    final ByteVector wBytes = ByteVector.fromMemorySegment(
                            BYTE_SPECIES_128, segment, groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    final ByteVector loBytes = wBytes.and((byte) 0xF);
                    final ByteVector hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    final int loIdx = loBase + c * 16;
                    final int hiIdx = hiBase + c * 16;

                    for (int p = 0; p < 2; p++) {
                        final int off = p * FLOAT_SPECIES.length();
                        final FloatVector loQ = (FloatVector) loBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector hiQ = (FloatVector) hiBytes.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);

                        acc = acc.add(loQ.mul(d1Vec).add(negM1Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, loIdx + off)));
                        acc = acc.add(hiQ.mul(d2Vec).add(negM2Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, hiIdx + off)));
                    }
                }
            }
            total += acc.reduceLanes(VectorOperators.ADD);
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 2));
            final long scalesOff = blockOffset + 4;
            final long qsOff = blockOffset + 16;
            final int inBase = inputOffset + blocks * blockSize;

            for (int i = 0; i < remainder; i++) {
                final int g = i / 64;
                final int inGroup = i % 64;
                final boolean isHigh = inGroup >= 32;
                final int l = isHigh ? inGroup - 32 : inGroup;
                final int subBlock = isHigh ? g * 2 + 1 : g * 2;

                final int sc = getScaleMinK4(subBlock, segment, scalesOff, false);
                final int m = getScaleMinK4(subBlock, segment, scalesOff, true);
                final byte qsByte = segment.get(ValueLayout.JAVA_BYTE, qsOff + g * 32 + l);
                final int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >>> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
                final float val = d * sc * quant - dmin * m;
                total += val * input[inBase + i];
            }
        }

        return total;
    }

    public static float dotQ5K(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        final int blockSize = 256;
        final int typeSize = 176;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 2));
            final long scalesOff = blockOffset + 4;
            final long qhOff = blockOffset + 16;
            final long qsOff = blockOffset + 48;
            final int inBase = inputOffset + b * blockSize;

            final ByteVector qh0 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, qhOff, ByteOrder.LITTLE_ENDIAN);
            final ByteVector qh1 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, qhOff + 16L, ByteOrder.LITTLE_ENDIAN);

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

            for (int g = 0; g < 4; g++) {
                final float d1 = d * getScaleMinK4(g * 2, segment, scalesOff, false);
                final float negM1 = -(dmin * getScaleMinK4(g * 2, segment, scalesOff, true));
                final float d2 = d * getScaleMinK4(g * 2 + 1, segment, scalesOff, false);
                final float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, segment, scalesOff, true));

                final FloatVector d1Vec = FloatVector.broadcast(FLOAT_SPECIES, d1);
                final FloatVector negM1Vec = FloatVector.broadcast(FLOAT_SPECIES, negM1);
                final FloatVector d2Vec = FloatVector.broadcast(FLOAT_SPECIES, d2);
                final FloatVector negM2Vec = FloatVector.broadcast(FLOAT_SPECIES, negM2);

                final int qhBitPosLo = 2 * g;
                final int qhBitPosHi = qhBitPosLo + 1;
                final long groupQsOff = qsOff + (long) g * 32;

                for (int c = 0; c < 2; c++) {
                    final int loBase = inBase + g * 64 + c * 16;
                    final int hiBase = inBase + g * 64 + 32 + c * 16;

                    final ByteVector wBytes = ByteVector.fromMemorySegment(
                            BYTE_SPECIES_128, segment, groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    ByteVector loQ = wBytes.and((byte) 0xF);
                    ByteVector hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    final ByteVector qhBytes = (c == 0) ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));

                    for (int p = 0; p < 2; p++) {
                        final int off = p * FLOAT_SPECIES.length();
                        final FloatVector loQf = (FloatVector) loQ.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector hiQf = (FloatVector) hiQ.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);

                        acc = acc.add(loQf.mul(d1Vec).add(negM1Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, loBase + off)));
                        acc = acc.add(hiQf.mul(d2Vec).add(negM2Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, hiBase + off)));
                    }
                }
            }
            total += acc.reduceLanes(VectorOperators.ADD);
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 2));
            final long scalesOff = blockOffset + 4;
            final long qhOff = blockOffset + 16;
            final long qsOff = blockOffset + 48;
            final int inBase = inputOffset + blocks * blockSize;

            for (int i = 0; i < remainder; i++) {
                final int g = i / 64;
                final int inGroup = i % 64;
                final boolean isHigh = inGroup >= 32;
                final int l = isHigh ? inGroup - 32 : inGroup;
                final int subBlock = isHigh ? g * 2 + 1 : g * 2;

                final int sc = getScaleMinK4(subBlock, segment, scalesOff, false);
                final int m = getScaleMinK4(subBlock, segment, scalesOff, true);
                final byte qsByte = segment.get(ValueLayout.JAVA_BYTE, qsOff + g * 32 + l);
                final int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >>> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
                final int qhBitPos = isHigh ? 2 * g + 1 : 2 * g;
                final int qhBit = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qhOff + l)) >>> qhBitPos) & 1;
                final int quant = nibble | (qhBit << 4);
                final float val = d * sc * quant - dmin * m;
                total += val * input[inBase + i];
            }
        }

        return total;
    }

    public static float dotQ2K(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        final int blockSize = 256;
        final int typeSize = 84;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final long scalesOff = blockOffset;
            final long qsOff = blockOffset + 16;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 80));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 82));
            final int inBase = inputOffset + b * blockSize;

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

            for (int n = 0; n < 2; n++) {
                final long qBase = qsOff + (long) n * 32;
                final ByteVector q0 = ByteVector.fromMemorySegment(
                        BYTE_SPECIES_128, segment, qBase, ByteOrder.LITTLE_ENDIAN);
                final ByteVector q1 = ByteVector.fromMemorySegment(
                        BYTE_SPECIES_128, segment, qBase + 16L, ByteOrder.LITTLE_ENDIAN);

                for (int j = 0; j < 4; j++) {
                    final int is0 = n * 8 + j * 2;
                    final int is1 = is0 + 1;
                    final byte sc0 = segment.get(ValueLayout.JAVA_BYTE, scalesOff + is0);
                    final byte sc1 = segment.get(ValueLayout.JAVA_BYTE, scalesOff + is1);
                    final float dl0 = d * (Byte.toUnsignedInt(sc0) & 0xF);
                    final float ml0 = dmin * (Byte.toUnsignedInt(sc0) >>> 4);
                    final float dl1 = d * (Byte.toUnsignedInt(sc1) & 0xF);
                    final float ml1 = dmin * (Byte.toUnsignedInt(sc1) >>> 4);

                    final int shift = j * 2;
                    final ByteVector v0 = q0.lanewise(VectorOperators.LSHR, shift).and((byte) 3);
                    final ByteVector v1 = q1.lanewise(VectorOperators.LSHR, shift).and((byte) 3);

                    final FloatVector dl0Vec = FloatVector.broadcast(FLOAT_SPECIES, dl0);
                    final FloatVector negMl0Vec = FloatVector.broadcast(FLOAT_SPECIES, -ml0);
                    final FloatVector dl1Vec = FloatVector.broadcast(FLOAT_SPECIES, dl1);
                    final FloatVector negMl1Vec = FloatVector.broadcast(FLOAT_SPECIES, -ml1);

                    final int idx0 = inBase + n * 128 + j * 32;
                    final int idx1 = idx0 + 16;

                    for (int p = 0; p < 2; p++) {
                        final int off = p * FLOAT_SPECIES.length();
                        final FloatVector vf0 = (FloatVector) v0.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector vf1 = (FloatVector) v1.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);

                        acc = acc.add(vf0.mul(dl0Vec).add(negMl0Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, idx0 + off)));
                        acc = acc.add(vf1.mul(dl1Vec).add(negMl1Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, idx1 + off)));
                    }
                }
            }
            total += acc.reduceLanes(VectorOperators.ADD);
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final long scalesOff = blockOffset;
            final long qsOff = blockOffset + 16;
            final float d = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 80));
            final float dmin = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 82));
            final int inBase = inputOffset + blocks * blockSize;

            for (int i = 0; i < remainder; i++) {
                final int n = i / 128;
                final int rem128 = i % 128;
                final int j = rem128 / 32;
                final int inSub32 = rem128 % 32;
                final boolean isSecondHalf = inSub32 >= 16;
                final int l = isSecondHalf ? inSub32 - 16 : inSub32;
                final int is = n * 8 + j * 2 + (isSecondHalf ? 1 : 0);
                final byte sc = segment.get(ValueLayout.JAVA_BYTE, scalesOff + is);
                final float dl = d * (Byte.toUnsignedInt(sc) & 0xF);
                final float ml = dmin * (Byte.toUnsignedInt(sc) >>> 4);
                final int shift = j * 2;
                final long qByteOff = qsOff + (long) n * 32 + (isSecondHalf ? 16 : 0) + l;
                final int qVal = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qByteOff)) >>> shift) & 3;
                final float val = dl * qVal - ml;
                total += val * input[inBase + i];
            }
        }

        return total;
    }

    public static float dotQ3K(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        final int blockSize = 256;
        final int typeSize = 110;
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        float total = 0.0f;

        final int kmask1 = 0x03030303;
        final int kmask2 = 0x0f0f0f0f;

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final long hmOff = blockOffset;
            final long qsOff = blockOffset + 32;
            final long scalesOff = blockOffset + 96;
            final float d_all = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 108));
            final int inBase = inputOffset + b * blockSize;

            final int aux0 = segment.get(LE_INT, scalesOff);
            final int aux1 = segment.get(LE_INT, scalesOff + 4);
            final int aux2 = segment.get(LE_INT, scalesOff + 8);
            final int tmp = aux2;
            final int u3 = ((aux1 >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
            final int u2 = ((aux0 >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
            final int u1 = (aux1 & kmask2) | (((tmp >>> 2) & kmask1) << 4);
            final int u0 = (aux0 & kmask2) | (((tmp >>> 0) & kmask1) << 4);

            final ByteVector hm0 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, hmOff, ByteOrder.LITTLE_ENDIAN);
            final ByteVector hm1 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, hmOff + 16L, ByteOrder.LITTLE_ENDIAN);

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int is = 0;

            for (int n = 0; n < 2; n++) {
                final long qBase = qsOff + (long) n * 32;
                final ByteVector q0 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, qBase, ByteOrder.LITTLE_ENDIAN);
                final ByteVector q1 = ByteVector.fromMemorySegment(BYTE_SPECIES_128, segment, qBase + 16L, ByteOrder.LITTLE_ENDIAN);

                for (int j = 0; j < 4; j++) {
                    final int is0 = is++;
                    final int is1 = is++;
                    final int sc0Val = getScaleQ3K(is0, u0, u1, u2, u3);
                    final int sc1Val = getScaleQ3K(is1, u0, u1, u2, u3);
                    final float dl0 = d_all * ((sc0Val & 63) - 32);
                    final float dl1 = d_all * ((sc1Val & 63) - 32);

                    final int shift = j * 2;
                    final int bitPos = n * 4 + j;

                    final ByteVector qVal0 = q0.lanewise(VectorOperators.LSHR, shift).and((byte) 3);
                    final ByteVector qVal1 = q1.lanewise(VectorOperators.LSHR, shift).and((byte) 3);

                    final ByteVector hBit0 = hm0.lanewise(VectorOperators.LSHR, bitPos).and((byte) 1);
                    final ByteVector hBit1 = hm1.lanewise(VectorOperators.LSHR, bitPos).and((byte) 1);

                    final ByteVector offset0 = hBit0.sub((byte) 1).and((byte) 4);
                    final ByteVector offset1 = hBit1.sub((byte) 1).and((byte) 4);

                    final ByteVector quant0 = qVal0.sub(offset0);
                    final ByteVector quant1 = qVal1.sub(offset1);

                    final FloatVector dl0Vec = FloatVector.broadcast(FLOAT_SPECIES, dl0);
                    final FloatVector dl1Vec = FloatVector.broadcast(FLOAT_SPECIES, dl1);

                    final int idx0 = inBase + n * 128 + j * 32;
                    final int idx1 = idx0 + 16;

                    for (int p = 0; p < 2; p++) {
                        final int off = p * FLOAT_SPECIES.length();
                        final FloatVector qf0 = (FloatVector) quant0.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);
                        final FloatVector qf1 = (FloatVector) quant1.convertShape(VectorOperators.B2F, FLOAT_SPECIES, p);

                        acc = acc.add(qf0.mul(dl0Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, idx0 + off)));
                        acc = acc.add(qf1.mul(dl1Vec).mul(FloatVector.fromArray(FLOAT_SPECIES, input, idx1 + off)));
                    }
                }
            }
            total += acc.reduceLanes(VectorOperators.ADD);
        }

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final long hmOff = blockOffset;
            final long qsOff = blockOffset + 32;
            final long scalesOff = blockOffset + 96;
            final float d_all = fp16ToFloat(segment.get(LE_SHORT, blockOffset + 108));
            final int inBase = inputOffset + blocks * blockSize;

            final int aux0 = segment.get(LE_INT, scalesOff);
            final int aux1 = segment.get(LE_INT, scalesOff + 4);
            final int aux2 = segment.get(LE_INT, scalesOff + 8);
            final int tmp = aux2;
            final int u3 = ((aux1 >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
            final int u2 = ((aux0 >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
            final int u1 = (aux1 & kmask2) | (((tmp >>> 2) & kmask1) << 4);
            final int u0 = (aux0 & kmask2) | (((tmp >>> 0) & kmask1) << 4);

            for (int i = 0; i < remainder; i++) {
                final int n = i / 128;
                final int rem128 = i % 128;
                final int j = rem128 / 32;
                final int inSub32 = rem128 % 32;
                final boolean isSecondHalf = inSub32 >= 16;
                final int l = isSecondHalf ? inSub32 - 16 : inSub32;
                final int is = n * 8 + j * 2 + (isSecondHalf ? 1 : 0);

                final int scVal = getScaleQ3K(is, u0, u1, u2, u3);
                final float dl = d_all * ((scVal & 63) - 32);
                final int shift = j * 2;
                final int bitPos = n * 4 + j;
                final int m = 1 << bitPos;
                final long qByteOff = qsOff + (long) n * 32 + (isSecondHalf ? 16 : 0) + l;
                final int qVal = (Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, qByteOff)) >>> shift) & 3;
                final long hmByteOff = hmOff + (isSecondHalf ? 16 : 0) + l;
                final int hmVal = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, hmByteOff));
                final int hVal = (hmVal & m) != 0 ? 0 : 4;
                final float val = dl * (qVal - hVal);
                total += val * input[inBase + i];
            }
        }

        return total;
    }

    private static int getScaleQ3K(final int is, final int u0, final int u1, final int u2, final int u3) {
        if (is < 4) {
            return (u0 >>> (is * 8)) & 0xFF;
        } else if (is < 8) {
            return (u1 >>> ((is - 4) * 8)) & 0xFF;
        } else if (is < 12) {
            return (u2 >>> ((is - 8) * 8)) & 0xFF;
        } else {
            return (u3 >>> ((is - 12) * 8)) & 0xFF;
        }
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

    // Call ONCE per input activation vector, before looping over weight rows.
//    public static float[] precomputeBlockSums(float[] input, int inputOffset, int size) {
//        final int blockSize = GgmlType.QK1_0;
//        final int lanes = FLOAT_SPECIES.length();
//        final int blocks = (size + blockSize - 1) / blockSize;
//        final float[] sums = new float[blocks];
//        for (int b = 0; b < blocks; b++) {
//            final int inBase = inputOffset + b * blockSize;
//            final int count = Math.min(blockSize, size - b * blockSize);
//            FloatVector vec = FloatVector.zero(FLOAT_SPECIES);
//            int g = 0;
//            for (; g + lanes <= count; g += lanes) {
//                vec = vec.add(FloatVector.fromArray(FLOAT_SPECIES, input, inBase + g));
//            }
//            float total = vec.reduceLanes(VectorOperators.ADD);
//            for (; g < count; g++) total += input[inBase + g];
//            sums[b] = total;
//        }
//        return sums;
//    }
//
//    // Called once per weight row — no more redundant total-summing.
//    public static float dotQ1(final MemorySegment segment, final long elementOffset,
//                              final float[] input, final int inputOffset, final int size) {
//        final int blockSize = GgmlType.QK1_0;
//        final int typeSize = 2 + blockSize / 8;
//        final int blocks = size / blockSize;
//        final long startBlock = elementOffset / blockSize;
//        float total = 0.0f;
//
//        for (int b = 0; b < blocks; b++) {
//            final long blockOffset = (startBlock + b) * typeSize;
//            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
//            final int inBase = inputOffset + b * blockSize;
//            final long loBits = segment.get(LE_LONG, blockOffset + Short.BYTES);
//            final long hiBits = segment.get(LE_LONG, blockOffset + Short.BYTES + Long.BYTES);
//
//            float posSum = 0.0f;
//            for (int k = 0; k < blockSize / 8; k++) {
//                final int bits = (int) ((k < 8 ? loBits : hiBits) >>> ((k % 8) * 8)) & 0xFF;
//                final int base = inBase + k * 8;
//                for (int j = 0; j < 8; j++) {
//                    posSum += ((bits >>> j) & 1) * input[base + j];
//                }
//            }
//            total += d * (2.0f * posSum - blockSums[b]);   // blockTotal now free
//        }
//        return total;
//    }

    /**
     * Q1_0: 128 elements per block. Block layout is an fp16 scale {@code d} followed by
     * 16 bytes of bits; element j dequantizes to {@code bit_j ? d : -d}.
     *
     * <p>Because every weight is exactly {@code ±d}, the dot product is a sign-flipped sum of the
     * inputs. Each of the 16 bit-bytes selects an 8-lane {@code ±1} vector from {@link #Q1_SIGN_TABLE},
     * so the inner loop is {@code input.fma(signs, acc)} — one FMA per 8 weights, no per-element
     * scalar work, no separate pass to compute the block total. The 16 bit bytes arrive in two
     * unaligned little-endian long reads, and the per-block scale is folded into a global vector
     * accumulator so there is only one horizontal reduction per row.</p>
     *
     * <p>Assumes {@code FLOAT_SPECIES} has 8 lanes (256-bit), like the other kernels here, so that
     * one bit byte maps exactly onto one vector.</p>
     */
    public static float dotQ1(final MemorySegment segment, final long elementOffset,
                              final float[] input, final int inputOffset, final int size) {
        final int blockSize = GgmlType.QK1_0;           // 128
        final int typeSize = 18; //2 + GgmlType.QK1_0 / 8;    // 18
        final int blocks = size / blockSize;
        final long startBlock = elementOffset / blockSize;
        final float[] signs = Q1_SIGN_TABLE;

        FloatVector totalAcc = FloatVector.zero(FLOAT_SPECIES);

        for (int b = 0; b < blocks; b++) {
            final long blockOffset = (startBlock + b) * typeSize;
            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final int inBase = inputOffset + b * blockSize;

            final long loBits = segment.get(LE_LONG, blockOffset + Short.BYTES);
            final long hiBits = segment.get(LE_LONG, blockOffset + Short.BYTES + Long.BYTES);

            // Two accumulators to break the FMA dependency chain.
            FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
            FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);

            for (int half = 0; half < 2; half++) {
                long bits = (half == 0) ? loBits : hiBits;
                final int halfBase = inBase + (half << 6);
                for (int k = 0; k < 8; k += 2) {
                    final int s0 = ((int) bits & 0xFF) << 3;
                    final int s1 = ((int) (bits >>> 8) & 0xFF) << 3;
                    bits >>>= 16;
                    final int idx = halfBase + (k << 3);
                    acc0 = FloatVector.fromArray(FLOAT_SPECIES, input, idx)
                            .fma(FloatVector.fromArray(FLOAT_SPECIES, signs, s0), acc0);
                    acc1 = FloatVector.fromArray(FLOAT_SPECIES, input, idx + 8)
                            .fma(FloatVector.fromArray(FLOAT_SPECIES, signs, s1), acc1);
                }
            }

            totalAcc = acc0.add(acc1).fma(FloatVector.broadcast(FLOAT_SPECIES, d), totalAcc);
        }

        float total = totalAcc.reduceLanes(VectorOperators.ADD);

        final int remainder = size % blockSize;
        if (remainder > 0) {
            final long blockOffset = (startBlock + blocks) * typeSize;
            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
            final long loBits = segment.get(LE_LONG, blockOffset + Short.BYTES);
            final long hiBits = segment.get(LE_LONG, blockOffset + Short.BYTES + Long.BYTES);
            final int inBase = inputOffset + blocks * blockSize;

            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int i = 0;
            for (; i + 8 <= remainder; i += 8) {
                final int group = i >>> 3;
                final int pattern = (int) ((group < 8 ? loBits : hiBits) >>> ((group & 7) << 3)) & 0xFF;
                acc = FloatVector.fromArray(FLOAT_SPECIES, input, inBase + i)
                        .fma(FloatVector.fromArray(FLOAT_SPECIES, signs, pattern << 3), acc);
            }
            float tail = acc.reduceLanes(VectorOperators.ADD);
            for (; i < remainder; i++) {
                final int group = i >>> 3;
                final int pattern = (int) ((group < 8 ? loBits : hiBits) >>> ((group & 7) << 3)) & 0xFF;
                tail += (((pattern >>> (i & 7)) & 1) != 0 ? 1.0f : -1.0f) * input[inBase + i];
            }
            total += d * tail;
        }

        return total;
    }

//    /**
//     * Q1_0: 128 elements per block. Block layout is an fp16 scale {@code d} followed by
//     * 16 bytes of bits; element j dequantizes to {@code bit_j ? d : -d}.
//     *
//     * <p>Since every weight is exactly {@code ±d}, the block dot product reduces to
//     * {@code d * (2 * posSum - total)} where {@code total} sums all inputs in the block and
//     * {@code posSum} sums only the inputs whose bit is set. The 16 bit bytes are pulled from
//     * the segment with two unaligned little-endian long reads so the inner loop is pure
//     * scalar math (no per-byte foreign calls), and the bit test is branchless to avoid
//     * mispredicts on random quantized data.</p>
//     */
//    public static float dotQ1(final MemorySegment segment, final long elementOffset,
//                              final float[] input, final int inputOffset, final int size) {
//        final int blockSize = GgmlType.QK1_0;
//        final int typeSize = 2 + GgmlType.QK1_0 / 8;
//        final int lanes = FLOAT_SPECIES.length();
//        final int blocks = size / blockSize;
//        final long startBlock = elementOffset / blockSize;
//        float total = 0.0f;
//
//        for (int b = 0; b < blocks; b++) {
//            final long blockOffset = (startBlock + b) * typeSize;
//            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
//            final int inBase = inputOffset + b * blockSize;
//
//            // All 16 bit bytes in two unaligned little-endian long reads (byte k is the kth
//            // least-significant byte of loBits for k < 8 and of hiBits for k >= 8).
//            final long loBits = segment.get(LE_LONG, blockOffset + Short.BYTES);
//            final long hiBits = segment.get(LE_LONG, blockOffset + Short.BYTES + Long.BYTES);
//
//            // total: sum of all 128 inputs via SIMD.
//            FloatVector vecTotal = FloatVector.zero(FLOAT_SPECIES);
//            for (int g = 0; g < blockSize / lanes; g++) {
//                vecTotal = vecTotal.add(FloatVector.fromArray(FLOAT_SPECIES, input, inBase + g * lanes));
//            }
//            final float blockTotal = vecTotal.reduceLanes(VectorOperators.ADD);
//
//            // posSum: sum of inputs whose bit is set (branchless).
//            float posSum = 0.0f;
//            for (int k = 0; k < blockSize / 8; k++) {
//                final int bits = (int) ((k < 8 ? loBits : hiBits) >>> (k % 8) * Long.BYTES) & 0xFF;
//                final int base = inBase + k * 8;
//                for (int j = 0; j < 8; j++) {
//                    posSum += (((bits >>> j) & 1)) * input[base + j];
//                }
//            }
//
//            total += d * (2.0f * posSum - blockTotal);
//        }
//
//        final int remainder = size % blockSize;
//        if (remainder > 0) {
//            final long blockOffset = (startBlock + blocks) * typeSize;
//            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
//            final long loBits = segment.get(LE_LONG, blockOffset + Short.BYTES);
//            final long hiBits = segment.get(LE_LONG, blockOffset + Short.BYTES + Long.BYTES);
//            final int inBase = inputOffset + blocks * blockSize;
//
//            float posSum = 0.0f;
//            float blockTotal = 0.0f;
//            for (int i = 0; i < remainder; i++) {
//                final int bits = (int) ((i < 64 ? loBits : hiBits) >>> ((i / 8) % 8) * Long.BYTES) & 0xFF;
//                posSum += (((bits >>> (i % 8)) & 1)) * input[inBase + i];
//                blockTotal += input[inBase + i];
//            }
//            total += d * (2.0f * posSum - blockTotal);
//        }
//
//        return total;
//    }

    /**
     * Q1_0: 128 elements per block. Block layout is an fp16 scale {@code d} followed by
     * 16 bytes of bits; element j dequantizes to {@code bit_j ? d : -d}.
     */
//    public static float dotQ1(final MemorySegment segment, final long elementOffset,
//                              final float[] input, final int inputOffset, final int size) {
//        final int blockSize = GgmlType.QK1_0;
//        final int typeSize = 2 + GgmlType.QK1_0 / 8;
//        final int blocks = size / blockSize;
//        final long startBlock = elementOffset / blockSize;
//        float total = 0.0f;
//
//        for (int b = 0; b < blocks; b++) {
//            final long blockOffset = (startBlock + b) * typeSize;
//            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
//            final float negD = -d;
//            final int inBase = inputOffset + b * blockSize;
//            float blockSum = 0.0f;
//            for (int byteIndex = 0; byteIndex < GgmlType.QK1_0 / 8; byteIndex++) {
//                final int bits = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, blockOffset + Short.BYTES + byteIndex));
//                final int base = byteIndex * 8;
//                for (int bit = 0; bit < 8; bit++) {
//                    blockSum += (((bits >>> bit) & 1) != 0 ? d : negD) * input[inBase + base + bit];
//                }
//            }
//            total += blockSum;
//        }
//
//        final int remainder = size % blockSize;
//        if (remainder > 0) {
//            final long blockOffset = (startBlock + blocks) * typeSize;
//            final float d = Float.float16ToFloat(segment.get(LE_SHORT, blockOffset));
//            final float negD = -d;
//            final int inBase = inputOffset + blocks * blockSize;
//            for (int i = 0; i < remainder; i++) {
//                final int bits = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, blockOffset + Short.BYTES + (i / 8)));
//                total += (((bits >>> (i % 8)) & 1) != 0 ? d : negD) * input[inBase + i];
//            }
//        }
//
//        return total;
//    }

    public static float dotF32(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        float total = 0.0f;
        FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
        int i = 0;
        final int lanes = FLOAT_SPECIES.length();
        final long byteOffset = elementOffset * Float.BYTES;
        for (; i + lanes <= size; i += lanes) {
            final FloatVector wVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, segment, byteOffset + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            final FloatVector inVec = FloatVector.fromArray(FLOAT_SPECIES, input, inputOffset + i);
            acc = acc.add(wVec.mul(inVec));
        }
        total += acc.reduceLanes(VectorOperators.ADD);
        for (; i < size; i++) {
            total += segment.get(LE_FLOAT, byteOffset + (long) i * Float.BYTES) * input[inputOffset + i];
        }
        return total;
    }

    public static float dotF16(final MemorySegment segment, final long elementOffset,
                               final float[] input, final int inputOffset, final int size) {
        float total = 0.0f;
        final long byteOffset = elementOffset * Short.BYTES;
        for (int i = 0; i < size; i++) {
            final float w = Float.float16ToFloat(segment.get(LE_SHORT, byteOffset + (long) i * Short.BYTES));
            total += w * input[inputOffset + i];
        }
        return total;
    }

    public static float dotBF16(final MemorySegment segment, final long elementOffset,
                                final float[] input, final int inputOffset, final int size) {
        float total = 0.0f;
        final long byteOffset = elementOffset * Short.BYTES;
        for (int i = 0; i < size; i++) {
            final short bits = segment.get(LE_SHORT, byteOffset + (long) i * Short.BYTES);
            final float w = Float.intBitsToFloat((bits & 0xFFFF) << 16);
            total += w * input[inputOffset + i];
        }
        return total;
    }

    private static float fp16ToFloat(final short h) {
        final int bits = Short.toUnsignedInt(h);
        final int sign = (bits & 0x8000) << 16;
        final int exp = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x03FF;

        if (exp == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign);
            }
            int e = 127 - 15 + 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                e--;
            }
            mantissa &= 0x03FF;
            return Float.intBitsToFloat(sign | (e << 23) | (mantissa << 13));
        }
        if (exp == 0x1F) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (mantissa << 13));
        }
        return Float.intBitsToFloat(sign | ((exp + (127 - 15)) << 23) | (mantissa << 13));
    }
}
