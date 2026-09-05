package com.epicmonstrosity.brewference.transformer.rope;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LongRopeScalingTest {
    private static final float[] SHORT = {1.0f, 1.2f, 0.8f, 1.5f};
    private static final float[] LONG = {1.1f, 1.0f, 0.9f, 1.4f};
    private static final int HEAD_DIM = 8;
    private static final float THETA = 10_000.0f;
    private static final int ORIG_MAX_POS = 4096;
    private static final float ATTN = 1.05f;
    private static final int HALF = HEAD_DIM / 2;

    private static LongRopeScaling scaling() {
        return new LongRopeScaling(SHORT, LONG, HEAD_DIM, THETA, ORIG_MAX_POS, ATTN);
    }

    private static float[] reference(final float[] factors, final int pos, final boolean cosine) {
        final float[] out = new float[HALF];
        for (int i = 0; i < HALF; i++) {
            final double invFreq = 1.0 / (factors[i] * Math.pow(THETA, (2.0 * i) / HEAD_DIM));
            final double angle = pos * invFreq;
            out[i] = (float) ((cosine ? Math.cos(angle) : Math.sin(angle)) * ATTN);
        }
        return out;
    }

    @Test
    void usesShortFactorsWithinOriginalContext() {
        final LongRopeScaling scaling = scaling();
        final float[] cos = new float[HALF];
        final float[] sin = new float[HALF];
        scaling.computeCosSinForPosition(123, ORIG_MAX_POS, cos, sin);
        assertArrayEquals(reference(SHORT, 123, true), cos);
        assertArrayEquals(reference(SHORT, 123, false), sin);
    }

    @Test
    void usesLongFactorsBeyondOriginalContext() {
        final LongRopeScaling scaling = scaling();
        final float[] cos = new float[HALF];
        final float[] sin = new float[HALF];
        scaling.computeCosSinForPosition(123, ORIG_MAX_POS + 1, cos, sin);
        assertArrayEquals(reference(LONG, 123, true), cos);
        assertArrayEquals(reference(LONG, 123, false), sin);
    }

    @Test
    void useRopeFactorMatchesReferenceAndSelectsBySequenceLength() {
        final LongRopeScaling scaling = scaling();

        final RopeCache.Result shortResult = scaling.useRopeFactor(HEAD_DIM, 100, ORIG_MAX_POS);
        assertArrayEquals(reference(SHORT, 100, true), shortResult.cosines);
        assertArrayEquals(reference(SHORT, 100, false), shortResult.sines);

        final RopeCache.Result longResult = scaling.useRopeFactor(HEAD_DIM, 100, ORIG_MAX_POS + 1);
        assertArrayEquals(reference(LONG, 100, true), longResult.cosines);
        assertArrayEquals(reference(LONG, 100, false), longResult.sines);
    }

    @Test
    void returnsCachedResultForRepeatedPositionAndSequenceLength() {
        final LongRopeScaling scaling = scaling();
        final RopeCache.Result first = scaling.useRopeFactor(HEAD_DIM, 7, 10);
        final RopeCache.Result second = scaling.useRopeFactor(HEAD_DIM, 7, 10);
        assertSame(first, second);
        assertArrayEquals(reference(SHORT, 7, true), first.cosines);

        final RopeCache.Result other = scaling.useRopeFactor(HEAD_DIM, 8, 10);
        assertEquals(reference(SHORT, 8, true)[0], other.cosines[0], 1e-6f);
    }

    @Test
    void rejectsMismatchedFactorLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new LongRopeScaling(new float[]{1.0f}, LONG, HEAD_DIM, THETA, ORIG_MAX_POS, ATTN));
    }

    @Test
    void rejectsMismatchedHeadSize() {
        assertThrows(IllegalArgumentException.class, () -> scaling().useRopeFactor(16, 0, 0));
    }
}
