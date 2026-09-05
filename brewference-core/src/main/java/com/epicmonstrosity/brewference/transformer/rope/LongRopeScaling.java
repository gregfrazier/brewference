package com.epicmonstrosity.brewference.transformer.rope;

import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;

/**
 * LongRoPE scaling (needed for Phi-3-mini-128k)
 * Two per-dimension frequency correction arrays (short/long), selected by
 * current sequence length relative to the model's original training context,
 * plus a scalar attention factor applied to all cos/sin values.
 *
 * Inverse frequencies are precomputed once at construction time so the
 * per-position loop only performs multiply + cos/sin. The most recent
 * (position, sequence length) result is cached because every layer of a
 * single token requests the same values.
 */
public final class LongRopeScaling {

    private final double[] shortInvFreqs;
    private final double[] longInvFreqs;
    private final int headDim;
    private final int originalMaxPositionEmbeddings;
    private final float attentionFactor;

    private int cachedPosition = -1;
    private int cachedSeqLen = -1;
    private RopeCache.Result cachedResult;

    public LongRopeScaling(final float[] shortFactors,
                           final float[] longFactors,
                           final int headDim,
                           final float ropeTheta,
                           final int originalMaxPositionEmbeddings,
                           final float attentionFactor) {
        this(new FloatArrayTensor(shortFactors), new FloatArrayTensor(longFactors),
                headDim, ropeTheta, originalMaxPositionEmbeddings, attentionFactor);
    }

    public LongRopeScaling(final FloatTensor shortFactors,
                           final FloatTensor longFactors,
                           final int headDim,
                           final float ropeTheta,
                           final int originalMaxPositionEmbeddings,
                           final float attentionFactor) {
        final int halfDim = headDim / 2;
        if (shortFactors.elementCount() != halfDim || longFactors.elementCount() != halfDim) {
            throw new IllegalArgumentException(
                    "factor array length " + shortFactors.elementCount() +
                            " != headDim/2 (" + halfDim + ")");
        }
        this.headDim = headDim;
        this.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings;
        this.attentionFactor = attentionFactor;
        this.shortInvFreqs = precomputeInvFreqs(shortFactors, halfDim, headDim, ropeTheta);
        this.longInvFreqs = precomputeInvFreqs(longFactors, halfDim, headDim, ropeTheta);
    }

    private static double[] precomputeInvFreqs(final FloatTensor factors, final int halfDim,
                                               final int headDim, final float ropeTheta) {
        final double[] invFreqs = new double[halfDim];
        for (int i = 0; i < halfDim; i++) {
            invFreqs[i] = 1.0 / (factors.get(i) * Math.pow(ropeTheta, (2.0 * i) / headDim));
        }
        return invFreqs;
    }

    /** Single-position variant for incremental decode, avoids recomputing the whole table. */
    public void computeCosSinForPosition(final int pos, final int currentSeqLen, final float[] cosOut, final float[] sinOut) {
        final double[] invFreqs = currentSeqLen > originalMaxPositionEmbeddings ? longInvFreqs : shortInvFreqs;
        final float attn = attentionFactor;
        for (int i = 0; i < invFreqs.length; i++) {
            final double angle = pos * invFreqs[i];
            cosOut[i] = (float) (Math.cos(angle) * attn);
            sinOut[i] = (float) (Math.sin(angle) * attn);
        }
    }

    public RopeCache.Result useRopeFactor(final int headSize, final int position, final int promptLength) {
        if (headSize != headDim) {
            throw new IllegalArgumentException("headSize %d != headDim %d".formatted(headSize, headDim));
        }
        final int seqLength = Math.max(position, promptLength);
        if (position == cachedPosition && seqLength == cachedSeqLen) {
            return cachedResult;
        }
        final int half = headSize / 2;
        final float[] cosines = new float[half];
        final float[] sines = new float[half];
        computeCosSinForPosition(position, seqLength, cosines, sines);
        cachedPosition = position;
        cachedSeqLen = seqLength;
        cachedResult = new RopeCache.Result(cosines, sines);
        return cachedResult;
    }
}
