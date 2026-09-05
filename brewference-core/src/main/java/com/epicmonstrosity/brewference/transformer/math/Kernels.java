package com.epicmonstrosity.brewference.transformer.math;

import com.epicmonstrosity.brewference.tensor.FloatTensor;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.concurrent.ThreadLocalRandom;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class Kernels {
    private static final float ZERO = 0.0f;
    private static final float ONE = 1.0f;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private Kernels() {}

    /**
     * Applies head-wise root mean square (RMS) normalization to an output array using the specified weights,
     * layer index, and other parameters. Each head in the output array is normalized independently.
     *
     * @param output the array where the normalized values will be stored
     * @param weight the tensor containing the weights used to scale the normalized values
     * @param layer the layer index used to compute the starting offset in the weight array
     * @param size the number of heads to process
     * @param headSize the size of each head in the output array
     * @param epsilon a small positive constant added to the denominator during RMS normalization to prevent division by zero
     */
    public static void headWiseRmsNorm(final float[] output, final FloatTensor weight, final int layer, final int size,
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
     * @param weight the tensor containing the weights used to scale the normalized values
     * @param weightOffset the starting offset in the weight tensor to begin accessing weights
     * @param size the number of elements in the input and weight tensor to process
     * @param epsilon a small positive constant added to the denominator to prevent division by zero
     */
    public static void rmsNorm(final float[] output, final int outputOffset, final float[] input, final int inputOffset,
                               final FloatTensor weight, final int weightOffset, final int size, final float epsilon) {
        final float scale = inverseRootMeanSquare(input, inputOffset, size, epsilon);
        int i = 0;

        if (weight instanceof FloatArrayTensor arrayWeight) {
            final float[] weightValues = arrayWeight.toArray();
            for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
                final FloatVector values = FloatVector.fromArray(FLOAT_SPECIES, input, inputOffset + i);
                final FloatVector weights = FloatVector.fromArray(FLOAT_SPECIES, weightValues, weightOffset + i);
                values.mul(scale).mul(weights).intoArray(output, outputOffset + i);
            }
        } else if (weight instanceof MappedF32Tensor mappedWeight) {
            final MemorySegment weightData = mappedWeight.data();
            for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
                final FloatVector values = FloatVector.fromArray(FLOAT_SPECIES, input, inputOffset + i);
                final FloatVector weights = FloatVector.fromMemorySegment(FLOAT_SPECIES, weightData,
                        (long) (weightOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                values.mul(scale).mul(weights).intoArray(output, outputOffset + i);
            }
        }

        for (; i < size; i++) {
            output[outputOffset + i] = input[inputOffset + i] * scale * weight.get(weightOffset + i);
        }
    }

    /**
     * Applies root mean square (RMS) normalization to the input array and writes the result to the output array.
     * RMS normalization scales the input based on the root mean square of its elements and multiplies the result
     * by the provided weight tensor.
     *
     * @param output the output array where the normalized values will be stored
     * @param input the input array containing the values to be normalized
     * @param weight the weight tensor used to scale the normalized values
     * @param weightOffset the starting offset in the weight tensor
     * @param size the number of elements in the input and weight tensor to process
     * @param epsilon a small positive constant added to prevent division by zero during normalization
     */
    public static void rmsNorm(final float[] output, final float[] input, final FloatTensor weight,
                               final int weightOffset, final int size, final float epsilon) {
        rmsNorm(output, 0, input, 0, weight, weightOffset, size, epsilon);
    }

    private static float inverseRootMeanSquare(final float[] input, final int inputOffset,
                                               final int size, final float epsilon) {
        float sumOfSquares = ZERO;
        int i = 0;
        FloatVector vectorSum = FloatVector.zero(FLOAT_SPECIES);

        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            final FloatVector values = FloatVector.fromArray(FLOAT_SPECIES, input, inputOffset + i);
            vectorSum = vectorSum.add(values.mul(values));
        }
        sumOfSquares = vectorSum.reduceLanes(VectorOperators.ADD);
        for (; i < size; i++) {
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
        int i = 1;
        FloatVector vectorMax = FloatVector.broadcast(FLOAT_SPECIES, maxValue);

        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            vectorMax = vectorMax.max(FloatVector.fromArray(FLOAT_SPECIES, values, startPosition + i));
        }
        maxValue = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < size; i++) {
            final float value = values[startPosition + i];
            if (value > maxValue) {
                maxValue = value;
            }
        }

        float sum = ZERO;
        i = 0;

        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            final FloatVector exponent = FloatVector.fromArray(FLOAT_SPECIES, values, startPosition + i)
                    .sub(maxValue)
                    .lanewise(VectorOperators.EXP);
            exponent.intoArray(values, startPosition + i);
            for (int lane = 0; lane < FLOAT_SPECIES.length(); lane++) {
                sum += exponent.lane(lane);
            }
        }
        for (; i < size; i++) {
            final float exponent = (float) Math.exp(values[startPosition + i] - maxValue);
            values[startPosition + i] = exponent;
            sum += exponent;
        }

        final float inverseSum = ONE / sum;
        i = 0;

        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, values, startPosition + i).mul(inverseSum)
                    .intoArray(values, startPosition + i);
        }
        for (; i < size; i++) {
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
        int i = 0;
        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, accumulator, i)
                    .add(FloatVector.fromArray(FLOAT_SPECIES, addend, i))
                    .intoArray(accumulator, i);
        }
        for (; i < size; i++) {
            accumulator[i] += addend[i];
        }
    }

    /**
     * Adds a slice of {@code bias} into {@code values}.
     *
     * @param values array modified in-place
     * @param bias bias tensor
     * @param offset starting offset in {@code bias}
     * @param size number of elements to add
     */
    public static void addBias(final float[] values, final FloatTensor bias, final int offset, final int size) {
        int i = 0;
        if (bias instanceof FloatArrayTensor arrayBias) {
            final float[] biasValues = arrayBias.toArray();
            for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
                FloatVector.fromArray(FLOAT_SPECIES, values, i)
                        .add(FloatVector.fromArray(FLOAT_SPECIES, biasValues, offset + i))
                        .intoArray(values, i);
            }
        } else if (bias instanceof MappedF32Tensor mappedBias) {
            final MemorySegment biasData = mappedBias.data();
            for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
                final FloatVector biasValues = FloatVector.fromMemorySegment(FLOAT_SPECIES, biasData,
                        (long) (offset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                FloatVector.fromArray(FLOAT_SPECIES, values, i).add(biasValues).intoArray(values, i);
            }
        }
        for (; i < size; i++) {
            values[i] += bias.get(offset + i);
        }
    }

    public static void scaleEmbeddings(final float[] values, final int size) {
        final float embeddingScale = (float) Math.sqrt(size);
        int i = 0;

        for (; i + FLOAT_SPECIES.length() <= size; i += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, values, i).mul(embeddingScale).intoArray(values, i);
        }
        for (; i < size; i++) {
            values[i] *= embeddingScale;
        }
    }
}
