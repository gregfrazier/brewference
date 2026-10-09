package com.epicmonstrosity.brewference.transformer.ssm;

/** Immutable mathematical description of a Gated Delta Net block. */
public record GatedDeltaNetSpec(
        int modelDim, int innerSize, int convChannels, int groupCount, int stateSize,
        int timeStepRank, int convKernel, float epsilon,
        GatedDeltaNetLayout layout, GatedDeltaNetGatePolicy gatePolicy) {
    public GatedDeltaNetSpec {
        if (modelDim <= 0 || innerSize <= 0 || convChannels <= 0 || groupCount <= 0 || stateSize <= 0
                || timeStepRank <= 0 || convKernel <= 1) {
            throw new IllegalArgumentException("Gated Delta Net dimensions must be positive");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("Gated Delta Net epsilon must be finite and positive");
        }
        if (innerSize != Math.multiplyExact(timeStepRank, stateSize)) {
            throw new IllegalArgumentException("Gated Delta Net requires innerSize == timeStepRank * stateSize");
        }
        if (timeStepRank % groupCount != 0) {
            throw new IllegalArgumentException("Gated Delta Net requires timeStepRank to be a multiple of groupCount");
        }
        final int groupedWidth = Math.multiplyExact(groupCount, stateSize);
        if (convChannels < Math.addExact(Math.multiplyExact(2, groupedWidth), innerSize)) {
            throw new IllegalArgumentException("Gated Delta Net convolution channels must contain Q, K, and V");
        }
        if (layout == null) throw new NullPointerException("layout");
        if (gatePolicy == null) throw new NullPointerException("gatePolicy");
        if (layout.headValueDim() != stateSize || layout.headCount() != timeStepRank) {
            throw new IllegalArgumentException("Gated Delta Net layout does not match spec dimensions");
        }
    }

    public int headValueDim() { return stateSize; }
    public int groupedHeadWidth() { return Math.multiplyExact(groupCount, stateSize); }
}