package com.epicmonstrosity.brewference.transformer.ssm;

import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.tensor.Tensor;

import java.util.Objects;

/** Narrow per-layer tensor view consumed by the generic Gated Delta Net algorithm. */
public record GatedDeltaNetWeights(QuantizedTensor qkv, QuantizedTensor z, QuantizedTensor decay,
                                   QuantizedTensor write, QuantizedTensor output, Tensor convolution,
                                   Tensor decayBase, Tensor decayBias, Tensor norm) {
    public GatedDeltaNetWeights {
        Objects.requireNonNull(qkv, "qkv");
        Objects.requireNonNull(z, "z");
        Objects.requireNonNull(decay, "decay");
        Objects.requireNonNull(write, "write");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(convolution, "convolution");
        Objects.requireNonNull(decayBase, "decayBase");
        Objects.requireNonNull(decayBias, "decayBias");
        Objects.requireNonNull(norm, "norm");
    }

    public void validate(final GatedDeltaNetSpec spec) {
        require(qkv, (long) spec.modelDim() * spec.convChannels(), "qkv");
        require(z, (long) spec.modelDim() * spec.innerSize(), "z");
        require(decay, (long) spec.modelDim() * spec.timeStepRank(), "decay");
        require(write, (long) spec.modelDim() * spec.timeStepRank(), "write");
        require(output, (long) spec.innerSize() * spec.modelDim(), "output");
        require(convolution, (long) spec.convChannels() * spec.convKernel(), "convolution");
        require(decayBase, spec.timeStepRank(), "decayBase");
        require(decayBias, spec.timeStepRank(), "decayBias");
        require(norm, spec.stateSize(), "norm");
    }

    private static void require(final Tensor tensor, final long expected, final String name) {
        if (tensor.elementCount() < expected) {
            throw new IllegalArgumentException("Gated Delta Net " + name + " has " + tensor.elementCount()
                    + " elements; expected at least " + expected);
        }
    }
}