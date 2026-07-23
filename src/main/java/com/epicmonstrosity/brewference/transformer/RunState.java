package com.epicmonstrosity.brewference.transformer;

/**
 * Mutable state for the transformer forward pass.
 */
public final class RunState {
    public float[] x;  // activation at the current time stamp
    public float[] xb;  // residual branch activation
    public float[] xb2; // convenience buffer
    public float[] hb; // hidden dimension in the ffn
    public float[] hb2; // hidden dimension buffer 2
    public float[] q;  // Query
    public float[] k;  // Key
    public float[] v;  // Value
    public float[] att; // attention value buffer
    public float[] logits; // output
    public float[] key_cache;
    public float[] value_cache;
}
