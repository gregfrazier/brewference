package com.epicmonstrosity.brewference.transformer.ssm;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD kernels for the Gated Delta Net recurrence. The split mirrors {@code Kernels} /
 * {@code QuantizedKernels}: the kernel math lives here, the block code in {@link GatedDeltaNet}
 * reads as a step list.
 * <p>
 * The delta state is stored row-major per head as {@code state[stateBase + i * headValueDim + j]}
 * with {@code i} the key row and {@code j} the value column, so a {@code j}-block is a contiguous
 * run of floats and the update can be expressed as vector accumulations over that run:
 * <pre>
 *   sk[j] = sum_i S[i][j] * k[i]   ->  accumulator += S_row_i * k[i]
 *   S[i][j] += k[i] * d[j]         ->  S_row_i     += k[i] * dVec
 *   o[j]  = sum_i S[i][j] * q[i]   ->  accumulator += S_row_i * q[i]
 * </pre>
 * The outer-product write and the read-out both traverse rows of the already-decayed state, so they
 * fuse into one pass: two traversals of the state instead of the scalar version's three.
 * <p>
 * Loop order: the {@code j}-block loop is outer and the row loop inner, so each block needs exactly
 * one live accumulator vector. The alternative — rows outer with one accumulator per {@code j}-block
 * held in an array — was implemented and measured under the same 16-head parallel split and rejected:
 * 214–222 µs per token against 139–170 µs for the scalar loops and 72–76 µs for this order. At
 * {@code headValueDim = 128} on an 8-lane species it needs 16 live accumulators, which is the whole
 * register file, so HotSpot spills the accumulator array to memory on every row. State traffic is the
 * same either way: a block pass visits each element exactly once per pass, and two adjacent blocks
 * share a cache line. Numbers: {@code docs/perf-results.md}.
 */
final class SsmKernels {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private SsmKernels() { }

    /**
     * One head's delta-rule step: decay the carried state, read it against {@code k}, write the
     * residual {@code d = (v - Sᵀk) * write}, then read the updated state back against {@code q}.
     * Two passes over the head's {@code headValueDim x headValueDim} slice, each walking one
     * {@code j}-block at a time.
     * <p>
     * State slice layout: {@code stateBase + i * headValueDim + j} (i = key row, j = value column).
     * {@code q}, {@code k}, {@code v}, {@code d} and {@code out} are read/written at
     * {@code headBase = head * headValueDim}. {@code d} is scratch (the per-head slice is private to
     * the head, so heads never share it); the {@code sk} buffer is not needed — the first pass feeds
     * {@code d} straight out of the accumulator.
     * <p>
     * Allocates nothing: the two accumulators and the {@code d} block vector are locals.
     */
    static void applyDeltaRule(final float[] state, final int stateBase,
                               final float[] q, final float[] k, final float[] v,
                               final float[] d, final float[] out,
                               final int headBase, final int headValueDim,
                               final float decay, final float write) {
        final int lanes = SPECIES.length();
        final int fullBlocks = headValueDim / lanes;
        final int tailStart = fullBlocks * lanes;
        final int tailLanes = headValueDim - tailStart;
        // A head narrower than one vector is a real case (Qwen35NumericsTest runs headValueDim 1 and
        // 2), so the leftover columns are a masked block rather than a scalar loop: masked-off lanes
        // are never read or written, which also keeps a head's slice private to that head.
        final VectorMask<Float> tailMask = tailLanes == 0
                ? null
                : VectorMask.fromLong(SPECIES, (1L << tailLanes) - 1L);

        // Pass 1 — decay the state in place and read it against k; the accumulator is sk for this
        // block and becomes d without ever landing in a buffer.
        for (int column = 0; column < tailStart; column += lanes) {
            FloatVector accumulator = FloatVector.zero(SPECIES);
            for (int i = 0; i < headValueDim; i++) {
                final int index = stateBase + i * headValueDim + column;
                final FloatVector decayed = FloatVector.fromArray(SPECIES, state, index).mul(decay);
                decayed.intoArray(state, index);
                accumulator = accumulator.add(decayed.mul(k[headBase + i]));
            }
            FloatVector.fromArray(SPECIES, v, headBase + column)
                    .sub(accumulator)
                    .mul(write)
                    .intoArray(d, headBase + column);
        }
        if (tailLanes != 0) {
            FloatVector accumulator = FloatVector.zero(SPECIES);
            for (int i = 0; i < headValueDim; i++) {
                final int index = stateBase + i * headValueDim + tailStart;
                final FloatVector decayed = FloatVector.fromArray(SPECIES, state, index, tailMask).mul(decay);
                decayed.intoArray(state, index, tailMask);
                accumulator = accumulator.add(decayed.mul(k[headBase + i]));
            }
            FloatVector.fromArray(SPECIES, v, headBase + tailStart, tailMask)
                    .sub(accumulator)
                    .mul(write)
                    .intoArray(d, headBase + tailStart, tailMask);
        }

        // Pass 2 — outer product and read-out, fused: both traverse rows of the decayed state.
        // A zero key skips only the write; the row still contributes to the read-out, so the row is
        // skipped only when both factors are zero (the scalar version could skip on key alone because
        // its read-out was a separate loop).
        for (int column = 0; column < tailStart; column += lanes) {
            final FloatVector writeVector = FloatVector.fromArray(SPECIES, d, headBase + column);
            FloatVector accumulator = FloatVector.zero(SPECIES);
            for (int i = 0; i < headValueDim; i++) {
                final float key = k[headBase + i];
                final float query = q[headBase + i];
                if (key == 0.0f && query == 0.0f) {
                    continue;
                }
                final int index = stateBase + i * headValueDim + column;
                FloatVector updated = FloatVector.fromArray(SPECIES, state, index);
                if (key != 0.0f) {
                    updated = updated.add(writeVector.mul(key));
                    updated.intoArray(state, index);
                }
                accumulator = accumulator.add(updated.mul(query));
            }
            accumulator.intoArray(out, headBase + column);
        }
        if (tailLanes != 0) {
            final FloatVector writeVector = FloatVector.fromArray(SPECIES, d, headBase + tailStart, tailMask);
            FloatVector accumulator = FloatVector.zero(SPECIES);
            for (int i = 0; i < headValueDim; i++) {
                final float key = k[headBase + i];
                final float query = q[headBase + i];
                if (key == 0.0f && query == 0.0f) {
                    continue;
                }
                final int index = stateBase + i * headValueDim + tailStart;
                FloatVector updated = FloatVector.fromArray(SPECIES, state, index, tailMask);
                if (key != 0.0f) {
                    updated = updated.add(writeVector.mul(key));
                    updated.intoArray(state, index, tailMask);
                }
                accumulator = accumulator.add(updated.mul(query));
            }
            accumulator.intoArray(out, headBase + tailStart, tailMask);
        }
    }
}
