package com.epicmonstrosity.brewference.transformer.ssm;

import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.transformer.math.Kernels;

import java.util.stream.IntStream;

/** Applies the per-head delta rule and gated RMS normalization. */
public final class GatedDeltaNetRecurrence {
    private static final int PARALLEL_HEAD_THRESHOLD = 16;
    private GatedDeltaNetRecurrence() { }

    public static void apply(final GatedDeltaNetWorkspace workspace, final GatedDeltaNetState state,
                             final GatedDeltaNetSpec spec, final Tensor norm) {
        if (spec.timeStepRank() >= PARALLEL_HEAD_THRESHOLD) {
            IntStream.range(0, spec.timeStepRank()).parallel()
                    .forEach(head -> updateHead(head, workspace, state, spec, norm));
        } else {
            for (int head = 0; head < spec.timeStepRank(); head++) updateHead(head, workspace, state, spec, norm);
        }
    }

    private static void updateHead(final int head, final GatedDeltaNetWorkspace workspace,
                                   final GatedDeltaNetState state, final GatedDeltaNetSpec spec,
                                   final Tensor norm) {
        final int headBase = head * spec.headValueDim();
        SsmKernels.applyDeltaRule(state.recurrent(), spec.layout().stateOffset(head), workspace.q(), workspace.k(),
                workspace.v(), workspace.delta(), workspace.output(), headBase, spec.headValueDim(),
                workspace.decay()[head], workspace.write()[head]);
        Kernels.rmsNorm(workspace.output(), headBase, workspace.output(), headBase, norm, 0,
                spec.headValueDim(), spec.epsilon());
        for (int i = 0; i < spec.headValueDim(); i++) {
            workspace.output()[headBase + i] *= GatedDeltaNet.silu(workspace.z()[headBase + i]);
        }
    }
}