package com.epicmonstrosity.brewference.transformer.ssm;

/** Architecture-selected formulas for decay and write gates. */
public interface GatedDeltaNetGatePolicy {
    float decay(float projected, float bias, float base);
    float write(float projected);
}