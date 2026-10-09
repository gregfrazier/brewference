package com.epicmonstrosity.brewference.transformer.ssm;

import com.epicmonstrosity.brewference.transformer.math.Linear;

/** Gated Delta Net orchestration. */
public final class GatedDeltaNet {
    private GatedDeltaNet() { }

    public static void forward(final GatedDeltaNetWorkspace workspace, final GatedDeltaNetState state,
                               final GatedDeltaNetWeights weights, final GatedDeltaNetSpec spec) {
        weights.validate(spec);
        workspace.validate(spec);
        state.validate(spec);

        final float[] residual = workspace.residual();
        final float[] qkv = workspace.qkv();
        final float[] z = workspace.z();
        final float[] convolution = workspace.convolution();
        final float[] q = workspace.q();
        final float[] k = workspace.k();
        final float[] v = workspace.v();
        final float[] decay = workspace.decay();
        final float[] write = workspace.write();
        final float[] convState = state.convolution();

        Linear.matmul(qkv, residual, weights.qkv(), 0, spec.modelDim(), spec.convChannels());
        Linear.matmul(z, residual, weights.z(), 0, spec.modelDim(), spec.innerSize());
        for (int channel = 0; channel < spec.convChannels(); channel++) {
            final int weightBase = channel * spec.convKernel();
            float sum = 0.0f;
            for (int tap = 0; tap < spec.convKernel() - 1; tap++) {
                sum += weights.convolution().value(weightBase + tap)
                        * convState[spec.layout().convolutionOffset(tap, channel)];
            }
            sum += weights.convolution().value(weightBase + spec.convKernel() - 1) * qkv[channel];
            convolution[channel] = silu(sum);
        }
        final int historyRows = spec.convKernel() - 1;
        System.arraycopy(convState, spec.convChannels(), convState, 0, (historyRows - 1) * spec.convChannels());
        System.arraycopy(qkv, 0, convState, (historyRows - 1) * spec.convChannels(), spec.convChannels());

        for (int head = 0; head < spec.timeStepRank(); head++) {
            final int target = head * spec.headValueDim();
            copyNormalized(convolution, spec.layout().queryOffset(head), q, target, spec.headValueDim(),
                    (float) (1.0 / Math.sqrt(spec.headValueDim())), spec.epsilon());
            copyNormalized(convolution, spec.layout().keyOffset(head), k, target, spec.headValueDim(),
                    1.0f, spec.epsilon());
            System.arraycopy(convolution, spec.layout().valueOffset(head), v, target, spec.headValueDim());
        }

        Linear.matmul(decay, residual, weights.decay(), 0, spec.modelDim(), spec.timeStepRank());
        Linear.matmul(write, residual, weights.write(), 0, spec.modelDim(), spec.timeStepRank());
        for (int head = 0; head < spec.timeStepRank(); head++) {
            decay[head] = spec.gatePolicy().decay(decay[head], weights.decayBias().value(head),
                    weights.decayBase().value(head));
            write[head] = spec.gatePolicy().write(write[head]);
        }

        GatedDeltaNetRecurrence.apply(workspace, state, spec, weights.norm());
        Linear.matmul(residual, workspace.output(), weights.output(), 0, spec.innerSize(), spec.modelDim());
    }

    private static void copyNormalized(final float[] source, final int sourceOffset, final float[] target,
                                       final int targetOffset, final int size, final float extraScale,
                                       final float epsilon) {
        float sumOfSquares = 0.0f;
        for (int i = 0; i < size; i++) {
            final float value = source[sourceOffset + i];
            sumOfSquares += value * value;
        }
        final float scale = (float) (1.0 / Math.sqrt(sumOfSquares + epsilon)) * extraScale;
        for (int i = 0; i < size; i++) target[targetOffset + i] = source[sourceOffset + i] * scale;
    }

    public static float sigmoid(final float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    public static float softplus(final float value) {
        if (value > 20.0f) return value;
        if (value < -20.0f) return (float) Math.exp(value);
        return (float) Math.log1p(Math.exp(value));
    }

    public static float silu(final float value) { return value * sigmoid(value); }
}
