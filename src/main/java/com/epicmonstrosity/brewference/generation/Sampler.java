package com.epicmonstrosity.brewference.generation;

import java.util.PriorityQueue;
import java.util.Random;

public class Sampler {
    private final Random rng;

    public Sampler(final long seed) {
        this.rng = new Random(seed);
    }

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

    private static void applyTemperature(final float[] logits, final int vocabSize, final float temperature) {
        if (temperature == 1.0f) {
            return;
        }

        for (int index = 0; index < vocabSize; index++) {
            logits[index] /= temperature;
        }
    }

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

    private int sampleCandidate(final int[] candidateIndices, final float[] probabilities, final int retainedCandidateCount) {
        float retainedProbabilitySum = 0.0f;
        for (int index = 0; index < retainedCandidateCount; index++) {
            retainedProbabilitySum += probabilities[index];
        }

        final float selectionThreshold = rng.nextFloat() * retainedProbabilitySum;
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
