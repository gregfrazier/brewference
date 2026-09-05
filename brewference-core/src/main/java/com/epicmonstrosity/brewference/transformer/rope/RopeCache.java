package com.epicmonstrosity.brewference.transformer.rope;

public final class RopeCache {
    private RopeCache() {}
    public static class Result {
        public final float[] cosines;
        public final float[] sines;

        public Result(final float[] cosines, final float[] sines) {
            this.cosines = cosines;
            this.sines = sines;
        }
    }

    public static Result precomputeAngles(final int headSize, final float ropeFrequencyBase, final float position) {
        final int half = headSize / 2;
        final float[] cosines = new float[half];
        final float[] sines = new float[half];
        for (int pair = 0; pair < half; pair++) {
            final float inverseFrequency = (float) (1.0 / Math.pow(ropeFrequencyBase, (2 * pair) / (float) headSize));
            final float angle = position * inverseFrequency;
            cosines[pair] = (float) Math.cos(angle);
            sines[pair] = (float) Math.sin(angle);
        }
        return new Result(cosines, sines);
    }
}
