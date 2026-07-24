package com.epicmonstrosity.brewference.transformer.math;

import java.util.concurrent.ThreadLocalRandom;

public final class Kernels {
    private static final float ZERO = 0.0f;
    private static final float ONE = 1.0f;

    private Kernels() {}

    /**
     * Applies head-wise root mean square (RMS) normalization to an output array using the specified weights,
     * layer index, and other parameters. Each head in the output array is normalized independently.
     *
     * @param output the array where the normalized values will be stored
     * @param weight the array containing the weights used to scale the normalized values
     * @param layer the layer index used to compute the starting offset in the weight array
     * @param size the number of heads to process
     * @param headSize the size of each head in the output array
     * @param epsilon a small positive constant added to the denominator during RMS normalization to prevent division by zero
     */
    public static void headWiseRmsNorm(final float[] output, final float[] weight, final int layer, final int size,
                                       final int headSize, final float epsilon) {
        final int weightOffset = layer * headSize;

        for (int head = 0; head < size; head++) {
            final int headOffset = head * headSize;
            rmsNorm(output, headOffset, output, headOffset, weight, weightOffset, headSize, epsilon);
        }
    }

    /**
     * Applies root mean square (RMS) normalization to a subset of the input array and writes the
     * results to the output array. RMS normalization scales the input elements based on the inverse
     * of the root mean square of their values, then multiplies the scaled elements by the
     * corresponding weights.
     *
     * @param output the array where the normalized values will be stored
     * @param outputOffset the starting offset in the output array where the results will be written
     * @param input the input array containing the values to be normalized
     * @param inputOffset the starting offset in the input array to begin the normalization process
     * @param weight the array containing the weights used to scale the normalized values
     * @param weightOffset the starting offset in the weight array to begin accessing weights
     * @param size the number of elements in the input and weight arrays to process
     * @param epsilon a small positive constant added to the denominator to prevent division by zero
     */
    public static void rmsNorm(final float[] output, final int outputOffset, final float[] input, final int inputOffset,
                               final float[] weight, final int weightOffset, final int size, final float epsilon) {
        final float scale = inverseRootMeanSquare(input, inputOffset, size, epsilon);

        for (int i = 0; i < size; i++) {
            final int outputIndex = outputOffset + i;
            final int inputIndex = inputOffset + i;
            final int weightIndex = weightOffset + i;

            output[outputIndex] = input[inputIndex] * scale * weight[weightIndex];
        }
    }

    /**
     * Applies root mean square (RMS) normalization to the input array and writes the result to the output array.
     * RMS normalization scales the input based on the root mean square of its elements and multiplies the result
     * by the provided weight array.
     *
     * @param output the output array where the normalized values will be stored
     * @param input the input array containing the values to be normalized
     * @param weight the weight array used to scale the normalized values
     * @param weightOffset the starting offset in the weight array
     * @param size the number of elements in the input and weight arrays to process
     * @param epsilon a small positive constant added to prevent division by zero during normalization
     */
    public static void rmsNorm(final float[] output, final float[] input, final float[] weight,
                               final int weightOffset, final int size, final float epsilon) {
        rmsNorm(output, 0, input, 0, weight, weightOffset, size, epsilon);
    }

    private static float inverseRootMeanSquare(final float[] input, final int inputOffset,
                                               final int size, final float epsilon) {
        float sumOfSquares = ZERO;

        for (int i = 0; i < size; i++) {
            final float value = input[inputOffset + i];
            sumOfSquares += value * value;
        }

        final float meanSquare = sumOfSquares / size;
        return (float) (ONE / Math.sqrt(meanSquare + epsilon));
    }

    /**
     * Applies the softmax function to a subset of the input array. The softmax function
     * normalizes the input values into a probability distribution, ensuring that the
     * sum of the values within the specified range equals 1. The transformation is
     * performed in place, modifying the original array.
     *
     * @param values the array of float values to be normalized
     * @param startPosition the starting index within the array to apply the softmax
     * @param size the number of elements to include in the softmax calculation
     */
    public static void softMax(final float[] values, final int startPosition, final int size) {
        float maxValue = values[startPosition];

        for (int i = 1; i < size; i++) {
            final float value = values[startPosition + i];
            if (value > maxValue) {
                maxValue = value;
            }
        }

        float sum = ZERO;

        for (int i = 0; i < size; i++) {
            final int index = startPosition + i;
            final float exponent = (float) Math.exp(values[index] - maxValue);

            values[index] = exponent;
            sum += exponent;
        }

        final float inverseSum = ONE / sum;

        for (int i = 0; i < size; i++) {
            //values[startPosition + i] /= sum;
            values[startPosition + i] *= inverseSum;
        }
    }

    public static int sampleSimple(final float[] probabilities, final int size) {
        final float randomValue = ThreadLocalRandom.current().nextFloat();
        float cumulativeProbability = ZERO;

        for (int i = 0; i < size; i++) {
            cumulativeProbability += probabilities[i];

            if (randomValue < cumulativeProbability) {
                return i;
            }
        }

        return size - 1;
    }

    /**
     * Finds the index of the maximum value in the given array up to a specified length.
     *
     * @param values the array of float values to search
     * @param size the number of elements to consider
     * @return the index of the maximum value within the specified range
     */
    public static int argMax(final float[] values, final int size) {
        int maxIndex = 0;
        float maxProbability = values[0];

        for (int i = 1; i < size; i++) {
            if (values[i] > maxProbability) {
                maxProbability = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * Adds {@code addend} into {@code accumulator}.
     *
     * @param accumulator array modified in-place
     * @param addend array added to {@code accumulator}
     * @param size number of elements to add
     */
    public static void accum(final float[] accumulator, final float[] addend, final int size) {
        for (int i = 0; i < size; i++) {
            accumulator[i] += addend[i];
        }
    }

    /**
     * Adds a slice of {@code bias} into {@code values}.
     *
     * @param values array modified in-place
     * @param bias bias array
     * @param offset starting offset in {@code bias}
     * @param size number of elements to add
     */
    public static void addBias(final float[] values, final float[] bias, final int offset, final int size) {
        for (int i = 0; i < size; i++) {
            values[i] += bias[offset + i];
        }
    }

    public static void scaleEmbeddings(final float[] values, final int size) {
        final float embeddingScale = (float) Math.sqrt(size);

        for (int i = 0; i < size; i++) {
            values[i] *= embeddingScale;
        }
    }
}
