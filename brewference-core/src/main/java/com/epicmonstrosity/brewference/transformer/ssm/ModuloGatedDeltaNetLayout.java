package com.epicmonstrosity.brewference.transformer.ssm;

/** Q/K/V grouped-head layout used by the current Gated Delta Net contract. */
public final class ModuloGatedDeltaNetLayout implements GatedDeltaNetLayout {
    private final int groupCount, stateSize, headCount, convChannels, convKernel;

    public ModuloGatedDeltaNetLayout(final int groupCount, final int stateSize, final int headCount,
                                     final int convChannels, final int convKernel) {
        if (groupCount <= 0 || stateSize <= 0 || headCount <= 0 || convChannels <= 0 || convKernel <= 1
                || headCount % groupCount != 0) throw new IllegalArgumentException("Invalid Gated Delta Net layout dimensions");
        this.groupCount = groupCount; this.stateSize = stateSize; this.headCount = headCount;
        this.convChannels = convChannels; this.convKernel = convKernel;
    }
    public int headCount() { return headCount; }
    public int headValueDim() { return stateSize; }
    public int queryOffset(final int head) { return sourceHead(head) * stateSize; }
    public int keyOffset(final int head) { return groupCount * stateSize + sourceHead(head) * stateSize; }
    public int valueOffset(final int head) { return 2 * groupCount * stateSize + head * stateSize; }
    public int sourceHead(final int head) { return head % groupCount; }
    public int stateOffset(final int head) { return Math.multiplyExact(head, Math.multiplyExact(stateSize, stateSize)); }
    public int convolutionOffset(final int tap, final int channel) {
        if (tap < 0 || tap >= convKernel - 1 || channel < 0 || channel >= convChannels) throw new IndexOutOfBoundsException();
        return tap * convChannels + channel;
    }
}