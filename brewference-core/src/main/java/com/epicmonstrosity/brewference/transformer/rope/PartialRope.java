package com.epicmonstrosity.brewference.transformer.rope;

/**
 * Partial (a.k.a. sliced) RoPE: rotate only the first {@code nRot} dimensions of each head and pass
 * the remaining {@code headSize - nRot} dimensions through bit-identical.
 * <p>
 * Pairing is <b>split-half</b> (NeoX): dimension {@code i} pairs with {@code i + nRot/2}, exactly the
 * way {@link NonInterleavedRope} pairs a whole head. This is what HF {@code Qwen3NextAttention}
 * does — {@code partial_rotary_factor = 0.25} with {@code apply_rotary_pos_emb} calling
 * {@code rotate_half} on the slice {@code [..., :rotaryDim]} — and what llama.cpp does with
 * {@code ROPE_MODE_SPLIT} and {@code n_rot = rope.dimension_count}. Adjacent pairing
 * {@code (2i, 2i+1)} would only be correct if the GGUF's q/k rows had been permuted at conversion;
 * they are not.
 * <p>
 * Properties: at position 0 the head is unchanged (cos = 1, sin = 0); a 64-dim rotation slice yields
 * 32 frequency pairs, not 128.
 * <p>
 * {@code nRot <= 0} means "the model did not publish {@code rope.dimension_count}" and falls back to
 * rotating the whole head, so this class degrades to {@link NonInterleavedRope}'s behaviour rather
 * than silently rotating nothing.
 */
public final class PartialRope {
    private static final double DEFAULT_FREQUENCY_BASE = 1_000_000.0;

    /**
     * Reused cos/sin table so the per-token hot path allocates nothing. Two floats per pair:
     * {@code [0, half)} cosines, {@code [half, 2*half)} sines.
     */
    private static final ThreadLocal<float[]> TABLE = ThreadLocal.withInitial(() -> new float[0]);

    private PartialRope() { }

    /**
     * Rotation width clamped to an even number no larger than the head; 0/unset means the whole head.
     */
    public static int effectiveRotationWidth(final int headSize, final int nRot) {
        final int width = nRot <= 0 ? headSize : Math.min(nRot, headSize);
        return Math.max(0, width) & ~1;
    }

    public static void apply(final float[] vector,
                             final int headCount,
                             final int headSize,
                             final int nRot,
                             final double frequencyBase,
                             final float position) {
        final int rotationWidth = effectiveRotationWidth(headSize, nRot);
        if (vector == null || rotationWidth <= 0 || headCount <= 0) {
            return;
        }
        final int half = rotationWidth / 2;
        final float[] table = table(rotationWidth, frequencyBase, position);

        for (int head = 0; head < headCount; head++) {
            final int headOffset = head * headSize;
            for (int dimension = 0; dimension < half; dimension++) {
                final float cosine = table[dimension];
                final float sine = table[half + dimension];

                final int firstIndex = headOffset + dimension;
                final int secondIndex = firstIndex + half;

                final float first = vector[firstIndex];
                final float second = vector[secondIndex];

                vector[firstIndex] = (float) (first * cosine - second * sine);
                vector[secondIndex] = (float) (first * sine + second * cosine);
            }
        }
    }

    /**
     * The cos/sin pairs this rotation would use, in the same order as
     * {@link RopeCache#precomputeAngles} for a head of {@code nRot} dimensions. Exposed for tests
     * and for callers that want to cache tables per position.
     */
    public static RopeCache.Result precompute(final int headSize, final int nRot,
                                              final double frequencyBase, final float position) {
        final int rotationWidth = effectiveRotationWidth(headSize, nRot);
        final int half = rotationWidth / 2;
        final double base = frequencyBase > 0.0 ? frequencyBase : DEFAULT_FREQUENCY_BASE;
        final float[] cosines = new float[half];
        final float[] sines = new float[half];
        for (int pair = 0; pair < half; pair++) {
            final double inverseFrequency = 1.0 / Math.pow(base, (2.0 * pair) / rotationWidth);
            final double angle = position * inverseFrequency;
            cosines[pair] = (float) Math.cos(angle);
            sines[pair] = (float) Math.sin(angle);
        }
        return new RopeCache.Result(cosines, sines);
    }

    private static float[] table(final int rotationWidth, final double frequencyBase, final float position) {
        final int half = rotationWidth / 2;
        float[] current = TABLE.get();
        if (current.length < 2 * half) {
            current = new float[2 * half];
            TABLE.set(current);
        }
        fillTable(current, rotationWidth, frequencyBase, position);
        return current;
    }

    private static void fillTable(final float[] table, final int rotationWidth,
                                  final double frequencyBase, final float position) {
        final int half = rotationWidth / 2;
        final double base = frequencyBase > 0.0 ? frequencyBase : DEFAULT_FREQUENCY_BASE;
        for (int pair = 0; pair < half; pair++) {
            final double inverseFrequency = 1.0 / Math.pow(base, (2.0 * pair) / rotationWidth);
            final double angle = position * inverseFrequency;
            table[pair] = (float) Math.cos(angle);
            table[half + pair] = (float) Math.sin(angle);
        }
    }
}
