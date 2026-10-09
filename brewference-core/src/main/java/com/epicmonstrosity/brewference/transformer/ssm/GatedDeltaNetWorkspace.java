package com.epicmonstrosity.brewference.transformer.ssm;

import java.util.Objects;

/** Caller-owned token workspace, allocated once per inference session. */
public final class GatedDeltaNetWorkspace {
    private final float[] residual, qkv, z, convolution, q, k, v, decay, write, output, sk, delta;

    public GatedDeltaNetWorkspace(final GatedDeltaNetSpec spec) {
        this(new float[spec.modelDim()], new float[spec.convChannels()], new float[spec.innerSize()],
                new float[spec.convChannels()], new float[spec.innerSize()], new float[spec.innerSize()],
                new float[spec.innerSize()], new float[spec.timeStepRank()], new float[spec.timeStepRank()],
                new float[spec.innerSize()], new float[spec.innerSize()], new float[spec.innerSize()]);
    }

    public GatedDeltaNetWorkspace(final float[] residual, final float[] qkv, final float[] z,
                                  final float[] convolution, final float[] q, final float[] k, final float[] v,
                                  final float[] decay, final float[] write, final float[] output,
                                  final float[] sk, final float[] delta) {
        this.residual = Objects.requireNonNull(residual, "residual"); this.qkv = Objects.requireNonNull(qkv, "qkv");
        this.z = Objects.requireNonNull(z, "z"); this.convolution = Objects.requireNonNull(convolution, "convolution");
        this.q = Objects.requireNonNull(q, "q"); this.k = Objects.requireNonNull(k, "k"); this.v = Objects.requireNonNull(v, "v");
        this.decay = Objects.requireNonNull(decay, "decay"); this.write = Objects.requireNonNull(write, "write");
        this.output = Objects.requireNonNull(output, "output"); this.sk = Objects.requireNonNull(sk, "sk");
        this.delta = Objects.requireNonNull(delta, "delta");
    }

    public void validate(final GatedDeltaNetSpec spec) {
        require(residual, spec.modelDim(), "residual"); require(qkv, spec.convChannels(), "qkv");
        require(z, spec.innerSize(), "z"); require(convolution, spec.convChannels(), "convolution");
        require(q, spec.innerSize(), "q"); require(k, spec.innerSize(), "k"); require(v, spec.innerSize(), "v");
        require(decay, spec.timeStepRank(), "decay"); require(write, spec.timeStepRank(), "write");
        require(output, spec.innerSize(), "output"); require(sk, spec.innerSize(), "sk"); require(delta, spec.innerSize(), "delta");
    }

    private static void require(final float[] values, final int expected, final String name) {
        if (values.length < expected) throw new IllegalArgumentException("Gated Delta Net workspace " + name
                + " has " + values.length + " elements; expected at least " + expected);
    }

    public float[] residual() { return residual; } public float[] qkv() { return qkv; } public float[] z() { return z; }
    public float[] convolution() { return convolution; } public float[] q() { return q; } public float[] k() { return k; }
    public float[] v() { return v; } public float[] decay() { return decay; } public float[] write() { return write; }
    public float[] output() { return output; } public float[] sk() { return sk; } public float[] delta() { return delta; }
}