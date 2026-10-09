package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;
import com.epicmonstrosity.brewference.transformer.attention.AttentionEngine;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;
import com.epicmonstrosity.brewference.transformer.ffn.FfnActivation;
import com.epicmonstrosity.brewference.transformer.math.Kernels;
import com.epicmonstrosity.brewference.transformer.math.Linear;
import com.epicmonstrosity.brewference.transformer.rope.PartialRope;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNet;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetSpec;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetWeights;

import java.util.stream.IntStream;

/**
 * Qwen3.5 dense forward pass.
 * <p>
 * <ul>
 *   <li><b>Layers are not shape-uniform.</b> Every third block is a Gated Delta Net whose
 *       {@code attn_qkv} projects to convChannels, and every fourth block has a <i>gated</i>
 *       {@code attn_q} that is twice as wide as the query stream. Flat composite weights addressed
 *       as {@code layer * dim * width} are therefore meaningless here — every access goes through the
 *       per-layer arrays.</li>
 *   <li><b>There is a norm between the attention residual and the FFN.</b> {@code post_attention_norm}
 *       is applied after the attention residual is added, before the FFN's own pre-norm would be.</li>
 * </ul>
 * Full-attention layers additionally use partial RoPE: only {@code rope.dimension_count} of the
 * 256-wide head rotates, split-half paired, and the attention output is multiplied by
 * {@code sigmoid(gate)} per query element before {@code attn_output}.
 */
public class Qwen35Transformer implements TransformerGraph {
    private static final System.Logger LOGGER = System.getLogger(Qwen35Transformer.class.getName());

    private final Config config;
    private final AttentionEngine attentionEngine;
    private final GatedDeltaNetSpec gatedDeltaNetSpec;
    private QuantizedWeights adaptedWeights;
    private GatedDeltaNetWeights[] gatedDeltaNetWeights;
    private boolean postAttentionNormFallbackLogged;

    public Qwen35Transformer(final Config config,
                             final AttentionPattern attentionPattern) {
        this.config = config;
        this.attentionEngine = new AttentionEngine(attentionPattern, config);
        this.gatedDeltaNetSpec = Qwen35GatedDeltaNetAdapter.spec(config);
    }

    @Override
    public void forward(final int token, final int pos, final RunState runState, final QuantizedWeights weights) {
        if (!(runState instanceof final Qwen35RunState state)) {
            throw new IllegalStateException("qwen35 needs a Qwen35RunState (produced by Qwen35ModelRunner.allocateRunState), got %s"
                    .formatted(runState.getClass().getSimpleName()));
        }

        final int dim = config.getTransformerDimensions();
        final float epsilon = config.getLayerNormRMSEpsilon();

        weights.dequantizeRow(weights.tokenEmbeddingTable, token * dim, dim, state.x);

        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            final LayerContext layerContext = new LayerContext(pos, layer,
                    layer * config.getMaxSequenceLength() * config.getKeyValueDim(), runState.promptSize);

            Kernels.rmsNorm(state.xb, state.x, require(weights.rmsAttLayer, layer, "attn_norm.weight"),
                    0, dim, epsilon);

            if (config.isFullAttentionLayer(layer)) {
                fullAttention(layer, layerContext, state, weights);
            } else {
                GatedDeltaNet.forward(state.gatedDeltaNetWorkspace(), state.gatedDeltaNetState(layer),
                        gatedDeltaNetWeights(weights, layer), gatedDeltaNetSpec);
                Kernels.accum(state.x, state.xb, dim);
            }

            // post_attention_norm: after the attention residual, before the FFN.
            Kernels.rmsNorm(state.xb, state.x, postAttentionNorm(weights, layer), 0, dim, epsilon);

            final int hiddenDim = config.getHiddenDimensions();
            Linear.matmul(state.hb, state.xb, require(weights.w1Layer, layer, "ffn_gate.weight"), 0, dim, hiddenDim);
            Linear.matmul(state.hb2, state.xb, require(weights.w3Layer, layer, "ffn_up.weight"), 0, dim, hiddenDim);
            FfnActivation.swiGlu(state, config);
            Linear.matmul(state.xb, state.hb, require(weights.w2Layer, layer, "ffn_down.weight"), 0, hiddenDim, dim);
            Kernels.accum(state.x, state.xb, dim);
        }

        Kernels.rmsNorm(state.x, state.x, weights.rmsFinalWeight, 0, dim, epsilon);
        Linear.matmul(state.logits, state.x, weights.classifier, 0, dim, config.getVocabSize());
    }

    private GatedDeltaNetWeights gatedDeltaNetWeights(final QuantizedWeights weights, final int layer) {
        if (adaptedWeights != weights) {
            final GatedDeltaNetWeights[] adapted = new GatedDeltaNetWeights[config.getNumLayers()];
            for (int i = 0; i < adapted.length; i++) {
                if (!config.isFullAttentionLayer(i)) adapted[i] = Qwen35GatedDeltaNetAdapter.weights(weights, i);
            }
            adaptedWeights = weights;
            gatedDeltaNetWeights = adapted;
        }
        return gatedDeltaNetWeights[layer];
    }

    /** Package-private so the gated full-attention block can be tested without a full forward pass. */
    void fullAttention(final int layer, final LayerContext layerContext,
                               final Qwen35RunState state, final QuantizedWeights weights) {
        final int dim = config.getTransformerDimensions();
        final int headSize = config.getHeadSize();
        final int numHeads = config.getNumHeads();
        final int numKVHeads = config.getNumKVHeads();
        final int qDim = config.getQueryAttentionWidth();
        final int kvDim = config.getKeyValueDim();
        final int gatedQueryWidth = config.getGatedQueryWidth();
        final float epsilon = config.getLayerNormRMSEpsilon();

        final QuantizedTensor queryWeight = require(weights.wqLayer, layer, "attn_q.weight");
        if (queryWeight.elementCount() < (long) gatedQueryWidth * dim) {
            throw new IllegalStateException("qwen35 full-attention layer %d expects a gated attn_q of %d x %d elements, found %d"
                    .formatted(layer, gatedQueryWidth, dim, queryWeight.elementCount()));
        }

        // attn_q is gated: per head the rows are [headSize query | headSize gate].
        Linear.matmul(state.qGate, state.xb, queryWeight, 0, dim, gatedQueryWidth);
        for (int head = 0; head < numHeads; head++) {
            final int rowBase = head * 2 * headSize;
            System.arraycopy(state.qGate, rowBase, state.q, head * headSize, headSize);
            System.arraycopy(state.qGate, rowBase + headSize, state.attnGate, head * headSize, headSize);
        }

        Linear.matmul(state.k, state.xb, require(weights.wkLayer, layer, "attn_k.weight"), 0, dim, kvDim);
        Linear.matmul(state.v, state.xb, require(weights.wvLayer, layer, "attn_v.weight"), 0, dim, kvDim);

        // Per-head Q/K RMS norm. Offset 0: these are per-layer weight tensors, not concatenated composites.
        Kernels.headWiseRmsNorm(state.q, require(weights.rmsQLayer, layer, "attn_q_norm.weight"), 0, numHeads, headSize, epsilon);
        Kernels.headWiseRmsNorm(state.k, require(weights.rmsKLayer, layer, "attn_k_norm.weight"), 0, numKVHeads, headSize, epsilon);

        // Partial RoPE: rope.dimension_count of the head rotates, the rest passes through.
        final int ropeDimensions = config.getRopeDimCount();
        final double ropeFrequencyBase = config.getRopeFrequencyBase();
        PartialRope.apply(state.q, numHeads, headSize, ropeDimensions, ropeFrequencyBase, layerContext.getTokenPosition());
        PartialRope.apply(state.k, numKVHeads, headSize, ropeDimensions, ropeFrequencyBase, layerContext.getTokenPosition());

        attentionEngine.storeKeyValueInCache(layerContext, state);
        IntStream.range(0, numHeads).parallel()
                .forEach(head -> attentionEngine.attend(layerContext, config, state, head));

        // AttentionEngine wrote the per-head outputs into xb; gate them before the output projection.
        for (int i = 0; i < qDim; i++) {
            state.xb[i] *= GatedDeltaNet.sigmoid(state.attnGate[i]);
        }

        Linear.matmul(state.xb2, state.xb, require(weights.woLayer, layer, "attn_output.weight"), 0, qDim, dim);
        Kernels.accum(state.x, state.xb2, dim);
    }

    /**
     * Some exports name the post-attention norm {@code ffn_norm} instead, leaving
     * {@code post_attention_norm.weight} absent. Fall back to the FFN norm and say so once.
     */
    private Tensor postAttentionNorm(final QuantizedWeights weights, final int layer) {
        final Tensor postAttention = layerOrNull(weights.postAttLayer, layer);
        if (postAttention != null) {
            return postAttention;
        }
        final Tensor fallback = layerOrNull(weights.rmsFfnLayer, layer);
        if (fallback == null) {
            throw new IllegalStateException("qwen35 layer %d has neither post_attention_norm.weight nor ffn_norm.weight"
                    .formatted(layer));
        }
        if (!postAttentionNormFallbackLogged) {
            postAttentionNormFallbackLogged = true;
            LOGGER.log(System.Logger.Level.WARNING,
                    "qwen35: post_attention_norm.weight missing, falling back to ffn_norm.weight for every layer");
        }
        return fallback;
    }

    private static Tensor layerOrNull(final Tensor[] tensors, final int layer) {
        return tensors != null && layer < tensors.length ? tensors[layer] : null;
    }

    private static Tensor require(final Tensor[] tensors, final int layer, final String name) {
        final Tensor tensor = layerOrNull(tensors, layer);
        if (tensor == null) {
            throw new IllegalStateException("qwen35 layer %d is missing %s".formatted(layer, name));
        }
        return tensor;
    }

    private static QuantizedTensor require(final QuantizedTensor[] tensors, final int layer, final String name) {
        if (tensors == null || layer >= tensors.length || tensors[layer] == null) {
            throw new IllegalStateException("qwen35 layer %d is missing %s".formatted(layer, name));
        }
        return tensors[layer];
    }
}
