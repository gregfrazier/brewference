package com.epicmonstrosity.brewference.transformer.ssm;

/** Describes projection and state offsets without tying the algorithm to a checkpoint format. */
public interface GatedDeltaNetLayout {
    int headCount();
    int headValueDim();
    int queryOffset(int head);
    int keyOffset(int head);
    int valueOffset(int head);
    int sourceHead(int head);
    int stateOffset(int head);
    int convolutionOffset(int tap, int channel);
}