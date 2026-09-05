package com.epicmonstrosity.brewference.transformer.math;

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
        final long matrixElements = Math.multiplyExact((long) inputSize, outputSize);
        if (weightsOffset % weights.blockSize() == 0 && inputSize % weights.blockSize() == 0) {
            final QuantizedTensor mapped = weights.mappedSlice(weightsOffset, matrixElements);
            if (mapped instanceof QuantizedSegmentTensor segmentTensor) {
                QuantizedKernels.matmul(output, input, segmentTensor, 0, inputSize, outputSize);
                return;
            }
        }
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
