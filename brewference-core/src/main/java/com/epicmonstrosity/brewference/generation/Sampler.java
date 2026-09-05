package com.epicmonstrosity.brewference.generation;

import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

public class Sampler {

    public Sampler() { }

    /**
     * Samples a token using temperature scaling, top-k filtering, and top-p nucleus sampling.
     *
     * @param logits raw model logits; this array is modified when temperature is not {@code 1.0f}
     * @param vocabSize number of logits to consider
     * @param temperature sampling temperature; must be greater than zero
     * @param topK maximum number of highest-logit candidates to retain
     * @param topP cumulative probability mass retained from the top-k candidates
     * @return the sampled token ID
     */
    public int sample(final float[] logits, final int vocabSize, final float temperature, final int topK, final float topP) {
        validateParameters(logits, vocabSize, temperature, topK, topP);

        applyTemperature(logits, vocabSize, temperature);

        final int[] candidateIndices = findTopKCandidateIndices(logits, vocabSize, topK);
        final float[] probabilities = calculateSoftmax(logits, candidateIndices);
        final int retainedCandidateCount = findNucleusCutoff(probabilities, topP);

        return sampleCandidate(candidateIndices, probabilities, retainedCandidateCount);
    }

    /**
     * Validates the sampling parameters before any work is performed.
     */
    private static void validateParameters(final float[] logits, final int vocabSize, final float temperature, final int topK, final float topP) {
        if (vocabSize <= 0 || vocabSize > logits.length)
            throw new IllegalArgumentException("vocabSize must be between 1 and logits.length");
        if (temperature <= 0.0f)
            throw new IllegalArgumentException("temperature must be greater than zero");
        if (topK <= 0)
            throw new IllegalArgumentException("topK must be greater than zero");
        if (topP <= 0.0f || topP > 1.0f)
            throw new IllegalArgumentException("topP must be in the range (0, 1]");
    }

    /**
     * Scales the logits in place by dividing each value by the temperature.
     * Skipped when the temperature is {@code 1.0f}, which leaves the distribution unchanged.
     *
     * @param logits raw model logits; modified in place
     * @param vocabSize number of logits to scale
     * @param temperature sampling temperature, already validated as greater than zero
     */
    private static void applyTemperature(final float[] logits, final int vocabSize, final float temperature) {
        if (temperature == 1.0f) {
            return;
        }

        for (int index = 0; index < vocabSize; index++) {
            logits[index] /= temperature;
        }
    }

    /**
     * Finds the indices of the {@code topK} highest logits using a bounded min-heap.
     *
     * @param logits raw model logits (already temperature-scaled)
     * @param vocabSize number of logits to consider
     * @param topK maximum number of candidates to retain; capped at {@code vocabSize}
     * @return token indices sorted in descending order of logit value
     */
    private static int[] findTopKCandidateIndices(final float[] logits, final int vocabSize, final int topK) {
        final int candidateLimit = Math.min(topK, vocabSize);
        final PriorityQueue<Integer> lowestCandidateFirst = new PriorityQueue<>(candidateLimit,
                (a, b) -> Float.compare(logits[a], logits[b]));

        for (int tokenId = 0; tokenId < vocabSize; tokenId++) {
            if (lowestCandidateFirst.size() < candidateLimit) {
                lowestCandidateFirst.offer(tokenId);
            } else if (logits[tokenId] > logits[lowestCandidateFirst.peek()]) {
                lowestCandidateFirst.poll();
                lowestCandidateFirst.offer(tokenId);
            }
        }

        final int[] candidateIndices = new int[lowestCandidateFirst.size()];
        for (int index = candidateIndices.length - 1; index >= 0; index--) {
            candidateIndices[index] = lowestCandidateFirst.poll();
        }
        return candidateIndices;
    }

    /**
     * Computes the softmax probabilities over the given candidates, subtracting the
     * maximum logit first for numerical stability.
     *
     * @param logits raw model logits (already temperature-scaled)
     * @param candidateIndices indices into {@code logits}, sorted in descending order of logit value
     * @return normalized probabilities aligned with {@code candidateIndices}
     */
    private static float[] calculateSoftmax(final float[] logits, final int[] candidateIndices) {
        final float maximumLogit = logits[candidateIndices[0]];
        final float[] probabilities = new float[candidateIndices.length];
        float unnormalizedProbabilitySum = 0.0f;

        for (int index = 0; index < candidateIndices.length; index++) {
            final float probability = (float) Math.exp(logits[candidateIndices[index]] - maximumLogit);
            probabilities[index] = probability;
            unnormalizedProbabilitySum += probability;
        }

        for (int index = 0; index < probabilities.length; index++) {
            probabilities[index] /= unnormalizedProbabilitySum;
        }
        return probabilities;
    }

    /**
     * Finds the smallest prefix of the sorted probabilities whose cumulative mass
     * reaches the nucleus threshold.
     *
     * @param probabilities softmax probabilities in descending order
     * @param topP cumulative probability mass threshold in the range {@code (0, 1]}
     * @return number of leading candidates to retain; at least one even if no prefix reaches {@code topP}
     */
    private static int findNucleusCutoff(final float[] probabilities, final float topP) {
        float cumulativeProbability = 0.0f;

        for (int index = 0; index < probabilities.length; index++) {
            cumulativeProbability += probabilities[index];
            if (cumulativeProbability >= topP) {
                return index + 1;
            }
        }

        return probabilities.length;
    }

    /**
     * Draws a single token from the retained candidates using weighted random selection.
     *
     * @param candidateIndices token indices sorted in descending order of logit value
     * @param probabilities softmax probabilities aligned with {@code candidateIndices}
     * @param retainedCandidateCount number of leading candidates to sample from
     * @return the sampled token ID; falls back to the highest-logit candidate if the
     *         random threshold is not reached due to floating-point rounding
     */
    private int sampleCandidate(final int[] candidateIndices, final float[] probabilities, final int retainedCandidateCount) {
        float retainedProbabilitySum = 0.0f;
        for (int index = 0; index < retainedCandidateCount; index++) {
            retainedProbabilitySum += probabilities[index];
        }

        final float selectionThreshold = ThreadLocalRandom.current().nextFloat() * retainedProbabilitySum;
        float cumulativeProbability = 0.0f;

        for (int index = 0; index < retainedCandidateCount; index++) {
            cumulativeProbability += probabilities[index];
            if (selectionThreshold <= cumulativeProbability) {
                return candidateIndices[index];
            }
        }

        return candidateIndices[0];
    }
}
