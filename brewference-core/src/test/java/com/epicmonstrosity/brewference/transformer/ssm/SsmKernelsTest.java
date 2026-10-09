package com.epicmonstrosity.brewference.transformer.ssm;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Equivalence tests for {@link SsmKernels#applyDeltaRule} against an inline copy of the scalar loops
 * it replaced in {@code GatedDeltaNet.updateHead}.
 * <p>
 * Tolerance is {@code 1e-4 * max(1, |reference|)}: the vector path changes the summation order
 * (accumulate over rows per value-column instead of over columns per key-row) and fuses the
 * outer-product write with the read-out, so last-ulp drift is expected. The pinned hand-computed
 * values in {@code Qwen35NumericsTest} are the strict check; this file is the coverage check.
 */
class SsmKernelsTest {
    private static final long SEED = 42L;
    private static final float RELATIVE_TOLERANCE = 1e-4f;

    // ---------------------------------------------------------------------------------------
    // 1. Single-step equivalence across head widths
    // ---------------------------------------------------------------------------------------

    @Test
    void matchesScalarReferenceForEveryHeadWidth() {
        // 1 and 2 are below the vector length (masked tail only), 7 too, 16/128 are exact multiples
        // for common species, 17 and 130 leave a partial tail block.
        for (final int headValueDim : new int[]{1, 2, 7, 16, 17, 128, 130}) {
            final Fixture fixture = Fixture.random(headValueDim, 2, new Random(SEED));
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            assertClose("headValueDim=%d state".formatted(headValueDim), reference.state, fixture.state, 0, reference.state.length);
            assertClose("headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);
        }
    }

    // ---------------------------------------------------------------------------------------
    // 2. Multi-step: the state carries, so a wrong pass order or a missing decay shows up
    // ---------------------------------------------------------------------------------------

    @Test
    void matchesScalarReferenceAcrossEightSequentialSteps() {
        for (final int headValueDim : new int[]{1, 3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED));
            final Fixture reference = fixture.copy();

            for (int step = 0; step < 8; step++) {
                // Fresh q/k/v each step; the state is whatever the previous step left behind.
                final Random random = new Random(SEED + step);
                fillRandom(fixture.q, fixture.headBase, headValueDim, random);
                fillRandom(fixture.k, fixture.headBase, headValueDim, random);
                fillRandom(fixture.v, fixture.headBase, headValueDim, random);
                System.arraycopy(fixture.q, fixture.headBase, reference.q, reference.headBase, headValueDim);
                System.arraycopy(fixture.k, fixture.headBase, reference.k, reference.headBase, headValueDim);
                System.arraycopy(fixture.v, fixture.headBase, reference.v, reference.headBase, headValueDim);

                scalarReference(reference);
                applyKernel(fixture);

                assertClose("headValueDim=%d step=%d state".formatted(headValueDim, step),
                        reference.state, fixture.state, 0, reference.state.length);
                assertClose("headValueDim=%d step=%d out".formatted(headValueDim, step),
                        reference.out, fixture.out, 0, reference.out.length);
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // 3. Extremes
    // ---------------------------------------------------------------------------------------

    @Test
    void decayZeroReducesTheStateToThePureOuterProduct() {
        for (final int headValueDim : new int[]{3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED)).decay(0.0f);
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            assertClose("decay=0 headValueDim=%d state".formatted(headValueDim), reference.state, fixture.state, 0, reference.state.length);
            assertClose("decay=0 headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);

            // Semantics, not just parity: the prior state is gone, S[i][j] == k[i] * v[j] * write.
            for (int i = 0; i < headValueDim; i++) {
                for (int j = 0; j < headValueDim; j++) {
                    assertEquals(fixture.k[fixture.headBase + i] * fixture.v[fixture.headBase + j] * fixture.write,
                            fixture.state[fixture.stateBase + i * headValueDim + j], 1e-4f,
                            "S[%d][%d]".formatted(i, j));
                }
            }
        }
    }

    @Test
    void decayOneLeavesThePriorStateUndecayed() {
        for (final int headValueDim : new int[]{3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED)).decay(1.0f);
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            assertClose("decay=1 headValueDim=%d state".formatted(headValueDim), reference.state, fixture.state, 0, reference.state.length);
            assertClose("decay=1 headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);
        }
    }

    @Test
    void writeZeroLeavesTheStateAtExactlyTheDecayedPriorState() {
        for (final int headValueDim : new int[]{3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED)).write(0.0f);
            final float[] prior = fixture.state.clone();
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            // A single multiply per element: no summation, so no drift is acceptable at all.
            for (int i = 0; i < prior.length; i++) {
                assertEquals(prior[i] * fixture.decay, fixture.state[i], 0.0f, "state[" + i + "]");
            }
            assertClose("write=0 headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);
        }
    }

    @Test
    void zeroKeysContributeNoOuterProductButStillReadOutTheState() {
        for (final int headValueDim : new int[]{3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED));
            Arrays.fill(fixture.k, fixture.headBase, fixture.headBase + headValueDim, 0.0f);
            final float[] prior = fixture.state.clone();
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            assertClose("k=0 headValueDim=%d state".formatted(headValueDim), reference.state, fixture.state, 0, reference.state.length);
            assertClose("k=0 headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);

            // sk is zero, so nothing is written back: the state is just the decayed prior state...
            for (int i = 0; i < prior.length; i++) {
                assertEquals(prior[i] * fixture.decay, fixture.state[i], 0.0f, "state[" + i + "]");
            }
            // ...and the read-out still ran (o[j] = sum_i decayed S[i][j] * q[i]).
            for (int j = 0; j < headValueDim; j++) {
                double expected = 0.0;
                for (int i = 0; i < headValueDim; i++) {
                    expected += prior[i * headValueDim + j] * fixture.decay * fixture.q[fixture.headBase + i];
                }
                assertEquals(expected, fixture.out[fixture.headBase + j], 1e-4 * Math.max(1.0, Math.abs(expected)),
                        "out[" + j + "]");
            }
        }
    }

    @Test
    void zeroValuesWriteTheNegatedReadOfK() {
        for (final int headValueDim : new int[]{3, 128}) {
            final Fixture fixture = Fixture.random(headValueDim, 1, new Random(SEED));
            Arrays.fill(fixture.v, fixture.headBase, fixture.headBase + headValueDim, 0.0f);
            final Fixture reference = fixture.copy();

            scalarReference(reference);
            applyKernel(fixture);

            assertClose("v=0 headValueDim=%d state".formatted(headValueDim), reference.state, fixture.state, 0, reference.state.length);
            assertClose("v=0 headValueDim=%d out".formatted(headValueDim), reference.out, fixture.out, 0, reference.out.length);
        }
    }

    // ---------------------------------------------------------------------------------------
    // 4. Head isolation
    // ---------------------------------------------------------------------------------------

    @Test
    void updatingOneHeadNeverTouchesAnotherHeadsStateOrOutput() {
        // headValueDim 1 is the interesting case: a masked vector load/store spans the neighbouring
        // head's slice, so it must be masked off rather than clamped.
        for (final int headValueDim : new int[]{1, 2, 3, 128}) {
            final int headCount = 2;
            final int headToUpdate = 1;
            final Fixture fixture = Fixture.random(headValueDim, headCount, new Random(SEED));
            final float[] stateBefore = fixture.state.clone();
            final float[] outBefore = fixture.out.clone();

            SsmKernels.applyDeltaRule(fixture.state, headToUpdate * headValueDim * headValueDim,
                    fixture.q, fixture.k, fixture.v, fixture.d, fixture.out,
                    headToUpdate * headValueDim, headValueDim, fixture.decay, fixture.write);

            final int stateSlice = headValueDim * headValueDim;
            final int untouchedStateFrom = 0;
            final int untouchedStateTo = stateSlice;
            for (int i = untouchedStateFrom; i < untouchedStateTo; i++) {
                assertEquals(Float.floatToIntBits(stateBefore[i]), Float.floatToIntBits(fixture.state[i]),
                        "head 0 state[" + i + "] changed (headValueDim=" + headValueDim + ")");
            }
            for (int i = 0; i < headValueDim; i++) {
                assertEquals(Float.floatToIntBits(outBefore[i]), Float.floatToIntBits(fixture.out[i]),
                        "head 0 out[" + i + "] changed (headValueDim=" + headValueDim + ")");
            }
            // The scratch head slice of the untouched head must be untouched too.
            for (int i = 0; i < headValueDim; i++) {
                assertEquals(Float.floatToIntBits(fixture.dBefore[i]), Float.floatToIntBits(fixture.d[i]),
                        "head 0 d[" + i + "] changed (headValueDim=" + headValueDim + ")");
            }
        }
    }

    /** Run the kernel over every head of the fixture, the way GatedDeltaNet does. */
    private static void applyKernel(final Fixture f) {
        for (int head = 0; head < f.headCount; head++) {
            SsmKernels.applyDeltaRule(f.state, head * f.headValueDim * f.headValueDim,
                    f.q, f.k, f.v, f.d, f.out, head * f.headValueDim,
                    f.headValueDim, f.decay, f.write);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Scalar reference — the loops GatedDeltaNet.updateHead used to run
    // ---------------------------------------------------------------------------------------

    private static void scalarReference(final Fixture f) {
        final int headValueDim = f.headValueDim;
        final float[] sk = new float[headValueDim];

        for (int head = 0; head < f.headCount; head++) {
            final int stateBase = head * headValueDim * headValueDim;
            final int headBase = head * headValueDim;
            Arrays.fill(sk, 0.0f);

            for (int i = 0; i < headValueDim; i++) {
                final float key = f.k[headBase + i];
                final int row = stateBase + i * headValueDim;
                for (int j = 0; j < headValueDim; j++) {
                    final float value = f.state[row + j] * f.decay;
                    f.state[row + j] = value;
                    sk[j] += value * key;
                }
            }
            for (int i = 0; i < headValueDim; i++) {
                f.d[headBase + i] = (f.v[headBase + i] - sk[i]) * f.write;
            }
            for (int i = 0; i < headValueDim; i++) {
                final float key = f.k[headBase + i];
                if (key == 0.0f) {
                    continue;
                }
                final int row = stateBase + i * headValueDim;
                for (int j = 0; j < headValueDim; j++) {
                    f.state[row + j] += key * f.d[headBase + j];
                }
            }
            for (int j = 0; j < headValueDim; j++) {
                float accumulator = 0.0f;
                for (int i = 0; i < headValueDim; i++) {
                    accumulator += f.state[stateBase + i * headValueDim + j] * f.q[headBase + i];
                }
                f.out[headBase + j] = accumulator;
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Fixture + helpers
    // ---------------------------------------------------------------------------------------

    /** One state array with {@code headCount} heads plus the per-head q/k/v/d/out slices. */
    private static final class Fixture {
        final float[] state;
        final float[] q;
        final float[] k;
        final float[] v;
        final float[] d;
        final float[] out;
        final float[] dBefore;
        final int headValueDim;
        final int headCount;
        /** The head the kernel under test updates; the scalar reference updates all of them. */
        final int headBase;
        final int stateBase;
        float decay;
        float write;

        Fixture(final int headValueDim, final int headCount) {
            this.headValueDim = headValueDim;
            this.headCount = headCount;
            this.state = new float[headCount * headValueDim * headValueDim];
            this.q = new float[headCount * headValueDim];
            this.k = new float[headCount * headValueDim];
            this.v = new float[headCount * headValueDim];
            this.d = new float[headCount * headValueDim];
            this.out = new float[headCount * headValueDim];
            this.dBefore = new float[headCount * headValueDim];
            this.headBase = 0;
            this.stateBase = 0;
            this.decay = 0.7f;
            this.write = 0.6f;
        }

        static Fixture random(final int headValueDim, final int headCount, final Random random) {
            final Fixture fixture = new Fixture(headValueDim, headCount);
            fillRandom(fixture.state, 0, fixture.state.length, random);
            fillRandom(fixture.q, 0, fixture.q.length, random);
            fillRandom(fixture.k, 0, fixture.k.length, random);
            fillRandom(fixture.v, 0, fixture.v.length, random);
            fillRandom(fixture.d, 0, fixture.d.length, random);
            System.arraycopy(fixture.d, 0, fixture.dBefore, 0, fixture.d.length);
            // decay in (0, 1] and write in [0, 1], the ranges the block actually produces.
            fixture.decay = 0.05f + 0.95f * random.nextFloat();
            fixture.write = random.nextFloat();
            return fixture;
        }

        Fixture copy() {
            final Fixture copy = new Fixture(headValueDim, headCount);
            System.arraycopy(state, 0, copy.state, 0, state.length);
            System.arraycopy(q, 0, copy.q, 0, q.length);
            System.arraycopy(k, 0, copy.k, 0, k.length);
            System.arraycopy(v, 0, copy.v, 0, v.length);
            System.arraycopy(d, 0, copy.d, 0, d.length);
            System.arraycopy(out, 0, copy.out, 0, out.length);
            copy.decay = decay;
            copy.write = write;
            return copy;
        }

        Fixture decay(final float value) {
            this.decay = value;
            return this;
        }

        Fixture write(final float value) {
            this.write = value;
            return this;
        }
    }

    private static void fillRandom(final float[] target, final int from, final int to, final Random random) {
        for (int i = from; i < to; i++) {
            target[i] = random.nextFloat() * 2.0f - 1.0f;
        }
    }

    private static void assertClose(final String label, final float[] expected, final float[] actual,
                                    final int from, final int to) {
        float worstRelative = 0.0f;
        float worstAbsolute = 0.0f;
        int worstIndex = from;
        for (int i = from; i < to; i++) {
            final float delta = Math.abs(expected[i] - actual[i]);
            final float relative = delta / Math.max(1.0f, Math.abs(expected[i]));
            if (Float.isNaN(relative) || relative > worstRelative) {
                worstRelative = relative;
                worstAbsolute = delta;
                worstIndex = i;
            }
        }
        final int index = worstIndex;
        final float relative = worstRelative;
        final float absolute = worstAbsolute;
        assertTrue(relative <= RELATIVE_TOLERANCE, () ->
                "%s: worst delta at index %d expected=%f actual=%f absolute=%g relative=%g (tolerance %g)"
                        .formatted(label, index, expected[index], actual[index], absolute, relative, RELATIVE_TOLERANCE));
    }
}
