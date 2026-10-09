package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetGatePolicy;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetLayout;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetSpec;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetWeights;
import com.epicmonstrosity.brewference.transformer.ssm.ModuloGatedDeltaNetLayout;
import com.epicmonstrosity.brewference.transformer.ssm.SoftplusSigmoidGatePolicy;

/** Converts Qwen35 configuration and loader-facing tensors into generic GDN contracts. */
public final class Qwen35GatedDeltaNetAdapter {
    private static final GatedDeltaNetGatePolicy GATE_POLICY = new SoftplusSigmoidGatePolicy();

    private Qwen35GatedDeltaNetAdapter() { }

    public static GatedDeltaNetSpec spec(final Config config) {
        final GatedDeltaNetLayout layout = new ModuloGatedDeltaNetLayout(config.getGroupCount(),
                config.getStateSize(), config.getTimeStepRank(), config.getConvChannels(), config.getConvKernel());
        return new GatedDeltaNetSpec(config.getTransformerDimensions(), config.getInnerSize(),
                config.getConvChannels(), config.getGroupCount(), config.getStateSize(),
                config.getTimeStepRank(), config.getConvKernel(), config.getLayerNormRMSEpsilon(), layout, GATE_POLICY);
    }

    public static GatedDeltaNetWeights weights(final QuantizedWeights weights, final int layer) {
        if (layer < 0) {
            throw new IllegalArgumentException("qwen35 Gated Delta Net layer must be non-negative, got " + layer);
        }
        return new GatedDeltaNetWeights(
                require(weights.attnQkvLayer, layer, "attn_qkv.weight"),
                require(weights.attnGateLayer, layer, "attn_gate.weight"),
                require(weights.ssmAlphaLayer, layer, "ssm_alpha.weight"),
                require(weights.ssmBetaLayer, layer, "ssm_beta.weight"),
                require(weights.ssmOutLayer, layer, "ssm_out.weight"),
                require(weights.ssmConv1dLayer, layer, "ssm_conv1d.weight"),
                require(weights.ssmALayer, layer, "ssm_a"),
                require(weights.ssmDtBiasLayer, layer, "ssm_dt.bias"),
                require(weights.ssmNormLayer, layer, "ssm_norm.weight"));
    }

    private static QuantizedTensor require(final QuantizedTensor[] tensors, final int layer, final String name) {
        if (tensors == null || layer < 0 || layer >= tensors.length || tensors[layer] == null) {
            throw new IllegalStateException("qwen35 layer %d is missing %s".formatted(layer, name));
        }
        return tensors[layer];
    }

    private static Tensor require(final Tensor[] tensors, final int layer, final String name) {
        if (tensors == null || layer < 0 || layer >= tensors.length || tensors[layer] == null) {
            throw new IllegalStateException("qwen35 layer %d is missing %s".formatted(layer, name));
        }
        return tensors[layer];
    }
}