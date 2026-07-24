package com.epicmonstrosity.brewference.transformer.math;

import com.epicmonstrosity.brewference.gguf.QuantizedWeights;

import java.util.stream.IntStream;

public final class Linear {
    private static final int Q8_BLOCK_SIZE = 32;

    private Linear() {}

    /**
     * Performs a matrix multiplication operation between a quantized weight matrix and an input vector,
     * storing the result in the output vector. The input and output vectors consist of floating-point
     * values, while the weights are represented as quantized data blocks to optimize memory utilization.
     *
     * The computation is performed using quantized weights, corresponding scale factors, and the input
     * data. The weight matrix is represented in blocks, and the appropriate offset and size values are
     * used to ensure correct alignment and indexing within the matrix.
     *
     * This method parallelizes the computation for each output row to improve performance.
     *
     * @param output The array where the result of the matrix multiplication will be stored.
     *               Its size must be equal to the specified output size.
     * @param input The array containing the input vector values. Its size must match the input size
     *              specified in the method arguments.
     * @param weights The quantized weight matrix, represented as blocks of quantized data and their
     *                corresponding scale factors.
     * @param weightsOffset The offset within the quantized weight matrix that specifies the starting
     *                      position for the computation.
     * @param inputSize The size of the input vector. This value determines the number of columns in
     *                  the weight matrix.
     * @param outputSize The size of the output vector. This value determines the number of rows in
     *                   the weight matrix.
     *
     * @throws IllegalArgumentException If any of the input arguments are invalid, such as mismatched
     *                                  dimensions between the input and weight matrix, or if the
     *                                  output array does not have sufficient capacity.
     */
    public static void matmul(final float[] output, final float[] input, final QuantizedWeights.Q8Block weights,
                              final int weightsOffset, final int inputSize, final int outputSize) {
        validateMatmulArguments(output, input, weights);
        final int blocksPerRow = (inputSize + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
        final int firstScaleBlock = weightsOffset / Q8_BLOCK_SIZE;

        final int lastScaleIndex = firstScaleBlock + (outputSize - 1) * blocksPerRow + (blocksPerRow - 1);
        if (lastScaleIndex >= weights.scales.length) {
            throw new IllegalStateException(String.format(
                    "matmul scale index out of bounds (wOffset=%d, n=%d, d=%d)", weightsOffset, inputSize, outputSize));
        }

        if (outputSize < 256) {
            IntStream.range(0, outputSize).forEach(row ->
                    output[row] = computeRowDotProduct(input, weights, weightsOffset,
                            inputSize, outputSize, blocksPerRow, firstScaleBlock, row
                    )
            );
        } else {
            IntStream.range(0, outputSize).parallel().forEach(row ->
                    output[row] = computeRowDotProduct(input, weights, weightsOffset,
                            inputSize, outputSize, blocksPerRow, firstScaleBlock, row
                    )
            );
        }
    }

    private static void validateMatmulArguments(final float[] output, final float[] input, final QuantizedWeights.Q8Block weights) {
        if (weights == null)
            throw new IllegalArgumentException("Weight block cannot be null");
        if (weights.scales == null || weights.data == null)
            throw new IllegalArgumentException("Weight block scales and data must be initialized");
        if (input == null || output == null)
            throw new IllegalArgumentException("Input and output arrays cannot be null");
    }

    private static float computeRowDotProduct(final float[] input, final QuantizedWeights.Q8Block weights,
                                              final int weightsOffset, final int inputSize, final int outputSize,
                                              final int blocksPerRow, final int firstScaleBlock, final int row) {
        final byte[] data = weights.data;
        final float[] scales = weights.scales;
        float rowSum = 0.0f;
        final int rowScaleBase = firstScaleBlock + row * blocksPerRow;
        final int rowDataBase = weightsOffset + row * inputSize;

        for (int block = 0; block < blocksPerRow; block++) {
            final int blockInputOffset = block * Q8_BLOCK_SIZE;
            final int blockSize = Math.min(Q8_BLOCK_SIZE, inputSize - blockInputOffset);
            final int dataIndex = rowDataBase + blockInputOffset;

            float blockSum = 0.0f;
            for (int offset = 0; offset < blockSize; offset++) {
                blockSum += data[dataIndex + offset] * input[blockInputOffset + offset];
            }

            rowSum += blockSum * scales[rowScaleBase + block];
        }

        return rowSum;
    }

//    private static void validateScaleIndex(final int scaleIndex, final int row, final int block,
//                                           final int weightsOffset, final int inputSize,
//                                           final int outputSize, final int scaleCount) {
//        if (scaleIndex >= scaleCount) {
//            throw new IllegalStateException(String.format("matmul scale index out of bounds: row=%d block=%d (wOffset=%d, n=%d, d=%d)", row, block, weightsOffset, inputSize, outputSize));
//        }
//    }
//
//    private static float computeScaledBlockSum(final float[] input, final QuantizedWeights.Q8Block weights,
//                                               final int scaleIndex, final int dataIndex, final int inputIndex, final int blockSize) {
//        float blockSum = 0.0f;
//
//        for (int offset = 0; offset < blockSize; offset++) {
//            blockSum += weights.data[dataIndex + offset] * input[inputIndex + offset];
//        }
//
//        return blockSum * weights.scales[scaleIndex];
//    }
}
