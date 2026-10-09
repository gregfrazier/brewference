package com.epicmonstrosity.brewference.transformer.ssm;

import java.util.Objects;

/** Persistent convolution and recurrent state for exactly one Gated Delta Net layer. */
public final class GatedDeltaNetState {
    private final float[] convolution;
    private final float[] recurrent;

    public GatedDeltaNetState(final GatedDeltaNetSpec spec) {
        this(new float[Math.multiplyExact(spec.convKernel() - 1, spec.convChannels())],
                new float[Math.multiplyExact(spec.timeStepRank(), Math.multiplyExact(spec.stateSize(), spec.stateSize()))]);
    }

    public GatedDeltaNetState(final float[] convolution, final float[] recurrent) {
        this.convolution = Objects.requireNonNull(convolution, "convolution");
        this.recurrent = Objects.requireNonNull(recurrent, "recurrent");
    }

    public void validate(final GatedDeltaNetSpec spec) {
        require(convolution, Math.multiplyExact(spec.convKernel() - 1, spec.convChannels()), "convolution state");
        require(recurrent, Math.multiplyExact(spec.timeStepRank(), Math.multiplyExact(spec.stateSize(), spec.stateSize())), "recurrent state");
    }

    private static void require(final float[] values, final int expected, final String name) {
        if (values.length < expected) throw new IllegalArgumentException("Gated Delta Net " + name + " has "
                + values.length + " elements; expected at least " + expected);
    }

    public float[] convolution() { return convolution; }
    public float[] recurrent() { return recurrent; }
}