package com.epicmonstrosity.brewference.transformer.ssm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GatedDeltaNetContractsTest {
    @Test
    void specAndLayoutExposeTheRowMajorStateContract() {
        final GatedDeltaNetSpec spec = spec(2, 2, 2, 3);

        assertEquals(2, spec.headValueDim());
        assertEquals(0, spec.layout().stateOffset(0));
        assertEquals(4, spec.layout().stateOffset(1));
        assertEquals(0, spec.layout().queryOffset(2));
        assertEquals(2, spec.layout().keyOffset(2));
        assertEquals(8, spec.layout().valueOffset(2));
        assertEquals(9, spec.layout().convolutionOffset(1, 1));
    }

    @Test
    void workspaceAndStateRejectBuffersThatCannotHoldAWholeToken() {
        final GatedDeltaNetSpec spec = spec(2, 2, 2, 3);
        final GatedDeltaNetWorkspace workspace = new GatedDeltaNetWorkspace(spec);
        final GatedDeltaNetState state = new GatedDeltaNetState(spec);

        workspace.output()[0] = 7.0f;
        state.recurrent()[0] = 3.0f;
        assertEquals(7.0f, workspace.output()[0]);
        assertEquals(3.0f, state.recurrent()[0]);
        assertThrows(IllegalArgumentException.class, () -> new GatedDeltaNetState(new float[1], new float[1]).validate(spec));
        assertThrows(IllegalArgumentException.class, () -> new GatedDeltaNetWorkspace(
                new float[1], new float[2], new float[2], new float[2], new float[2], new float[2],
                new float[2], new float[1], new float[1], new float[2], new float[2], new float[2]).validate(spec));
    }

    @Test
    void specRejectsInconsistentHeadRelationships() {
        assertThrows(IllegalArgumentException.class, () -> new GatedDeltaNetSpec(2, 3, 6, 1, 2, 2, 3,
                1.0e-6f, new ModuloGatedDeltaNetLayout(1, 2, 2, 6, 3), new SoftplusSigmoidGatePolicy()));
    }

    private static GatedDeltaNetSpec spec(final int modelDim, final int stateSize,
                                          final int heads, final int convKernel) {
        final int groups = 1;
        final int innerSize = stateSize * heads;
        final int convChannels = 2 * groups * stateSize + innerSize;
        return new GatedDeltaNetSpec(modelDim, innerSize, convChannels, groups, stateSize, heads, convKernel,
                1.0e-6f, new ModuloGatedDeltaNetLayout(groups, stateSize, heads, convChannels, convKernel),
                new SoftplusSigmoidGatePolicy());
    }
}