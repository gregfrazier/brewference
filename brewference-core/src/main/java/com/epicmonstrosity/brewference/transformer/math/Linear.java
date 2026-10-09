package com.epicmonstrosity.brewference.transformer.math;

import com.epicmonstrosity.brewference.tensor.CompositeQuantizedTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;

import java.util.stream.IntStream;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class Linear {
    private Linear() {}

    /**
     * Performs a matrix multiplication operation between a quantized weight matrix and an input vector,
     * storing the result in the output vector.
     */
    public static void matmul(final float[] output, final float[] input, final QuantizedTensor weights,
                              final int weightsOffset, final int inputSize, final int outputSize) {
        validateMatmulArguments(output, input, weights, weightsOffset, inputSize, outputSize);
        if (weights instanceof QuantizedSegmentTensor segmentTensor) {
            QuantizedKernels.matmul(output, input, segmentTensor, weightsOffset, inputSize, outputSize);
            return;
        }
        if (weights instanceof CompositeQuantizedTensor compositeTensor) {
            matmulComposite(output, input, compositeTensor, weightsOffset, inputSize, outputSize);
            return;
        }
        matmulByValue(output, input, weights, weightsOffset, inputSize, outputSize);
    }

    /**
     * Matmul over a weight that is stitched together from several quantized parts (one part per layer).
     * <p>
     * The parts may use different GGML formats, so a single segment view of the whole matrix only exists
     * when the range lies inside one part and is aligned to that part's block size - the usual case, one
     * layer's weight being exactly one part. Otherwise the composite is walked in part-aligned runs so
     * each run still runs on its own block kernel: kernels derive {@code startBlock = elementOffset /
     * blockSize}, so a run must be aligned to the part's block size and cover whole rows. Rows that no
     * such run can cover are read element by element, which is the correctness net, not the fast path.
     */
    private static void matmulComposite(final float[] output, final float[] input,
                                        final CompositeQuantizedTensor weights,
                                        final int weightsOffset, final int inputSize, final int outputSize) {
        final long matrixElements = Math.multiplyExact((long) inputSize, outputSize);
        if (matrixElements == 0) {
            return;
        }
        final int alignment = weights.blockSize();

        if (weightsOffset % alignment == 0 && inputSize % alignment == 0) {
            final QuantizedTensor whole = weights.mappedSlice(weightsOffset, matrixElements);
            if (whole instanceof QuantizedSegmentTensor segment) {
                QuantizedKernels.matmul(output, input, segment, 0, inputSize, outputSize);
                return;
            }
        }

        final long end = weightsOffset + matrixElements;
        long cursor = weightsOffset;
        int row = 0;
        while (cursor < end) {
            final int partIndex = weights.partIndexFor(cursor);
            final int partBlockSize = weights.partBlockSize(partIndex);
            final long runElements = Math.min(end, weights.partElementEnd(partIndex)) - cursor;
            final boolean kernelFriendlyRun = cursor % partBlockSize == 0
                    && runElements % partBlockSize == 0
                    && runElements % inputSize == 0
                    && inputSize % partBlockSize == 0;
            if (kernelFriendlyRun) {
                final QuantizedTensor slice = weights.mappedSlice(cursor, runElements);
                if (slice instanceof QuantizedSegmentTensor segment) {
                    final int rows = (int) (runElements / inputSize);
                    QuantizedKernels.matmul(output, row, input, segment, 0, inputSize, rows);
                    row += rows;
                    cursor += runElements;
                    continue;
                }
            }
            output[row] = dotRow(input, weights, cursor, inputSize);
            row++;
            cursor += inputSize;
        }
    }

    private static void matmulByValue(final float[] output, final float[] input, final QuantizedTensor weights,
                                      final int weightsOffset, final int inputSize, final int outputSize) {
        final IntStream rows = IntStream.range(0, outputSize);
        (outputSize < 256 ? rows : rows.parallel()).forEach(row -> {
            float rowSum = 0.0f;
            final long rowOffset = (long) weightsOffset + (long) row * inputSize;
            for (int column = 0; column < inputSize; column++) {
                rowSum += weights.value(rowOffset + column) * input[column];
            }
            output[row] = rowSum;
        });
    }

    private static float dotRow(final float[] input, final QuantizedTensor weights,
                                final long rowOffset, final int inputSize) {
        float rowSum = 0.0f;
        for (int column = 0; column < inputSize; column++) {
            rowSum += weights.value(rowOffset + column) * input[column];
        }
        return rowSum;
    }

    private static void validateMatmulArguments(final float[] output, final float[] input,
                                                final QuantizedTensor weights,
                                                final int weightsOffset, final int inputSize, final int outputSize) {
        if (weights == null)
            throw new IllegalArgumentException("Weight block cannot be null");
        if (input == null || output == null)
            throw new IllegalArgumentException("Input and output arrays cannot be null");
        if (weightsOffset < 0 || inputSize < 0 || outputSize < 0 || input.length < inputSize || output.length < outputSize) {
            throw new IllegalArgumentException("Invalid matmul dimensions");
        }
        final long matrixElements;
        try {
            matrixElements = Math.multiplyExact((long) inputSize, outputSize);
        } catch (final ArithmeticException e) {
            throw new IllegalArgumentException("Matmul dimensions overflow", e);
        }
        if (matrixElements > weights.elementCount() - weightsOffset) {
            throw new IllegalStateException(String.format(
                    "matmul weights out of bounds (wOffset=%d, n=%d, d=%d)", weightsOffset, inputSize, outputSize));
        }
    }

}
