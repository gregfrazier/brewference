package com.epicmonstrosity.brewference.transformer.ssm;

/** Softplus decay and sigmoid write-gate policy used by the Qwen-compatible formulation. */
public final class SoftplusSigmoidGatePolicy implements GatedDeltaNetGatePolicy {
    public float decay(final float projected, final float bias, final float base) {
        return (float) Math.exp(GatedDeltaNet.softplus(projected + bias) * base);
    }
    public float write(final float projected) { return sigmoid(projected); }
    private static float sigmoid(final float value) { return (float) (1.0 / (1.0 + Math.exp(-value))); }
}