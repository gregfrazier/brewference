package com.epicmonstrosity.brewference.model.qwen2;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.gguf.QuantizedWeights;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.TransformerGraph;
import com.epicmonstrosity.brewference.transformer.attention.AttentionEngine;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;
import com.epicmonstrosity.brewference.transformer.ffn.FfnActivation;
import com.epicmonstrosity.brewference.transformer.math.Kernels;
import com.epicmonstrosity.brewference.transformer.math.Linear;
import com.epicmonstrosity.brewference.transformer.rope.NonInterleavedRope;

import java.util.stream.IntStream;

public class Qwen2Transformer implements TransformerGraph {
    private final Config config;
    private final AttentionEngine attentionEngine;

    public Qwen2Transformer(final Config config,
                            final AttentionPattern attentionPattern) {
        this.config = config;
        this.attentionEngine = new AttentionEngine(attentionPattern, config);
    }

    @Override
    public void forward(final int token, final int pos, final RunState runState, final QuantizedWeights weights) {
        final int dim = config.getTransformerDimensions();

        weights.dequantizeRow(weights.tokenEmbeddingTable, token * dim, dim, runState.x);

        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            final LayerContext layerContext = new LayerContext(pos, layer, layer * config.getMaxSequenceLength() * config.getKeyValueDim());

            // Attention RMS Normalization
            Kernels.rmsNorm(runState.xb, runState.x, weights.rmsAttWeight, layer * dim, dim, config.getLayerNormRMSEpsilon());

            // QKV Projections
            Linear.matmul(runState.q, runState.xb, weights.wq, layer * dim * dim, dim, dim);
            Linear.matmul(runState.k, runState.xb, weights.wk, layer * dim * config.getKeyValueDim(), dim, config.getKeyValueDim());
            Linear.matmul(runState.v, runState.xb, weights.wv, layer * dim * config.getKeyValueDim(), dim, config.getKeyValueDim());

            // Bias
            Kernels.addBias(runState.q, weights.qBias, layer * config.getQueryAttentionWidth(), config.getQueryAttentionWidth());
            Kernels.addBias(runState.k, weights.kBias, layer * config.getKeyValueDim(), config.getKeyValueDim());
            Kernels.addBias(runState.v, weights.vBias, layer * config.getKeyValueDim(), config.getKeyValueDim());

            // Rope
            NonInterleavedRope.apply(runState, config, layerContext, () -> 1_000_000.0f);

            attentionEngine.storeKeyValueInCache(layerContext, runState);

            IntStream.range(0, config.getNumHeads()).parallel().forEach(head -> attentionEngine.attend(layerContext, config, runState, head));

            // Attention output projection
            Linear.matmul(runState.xb2, runState.xb, weights.wo, layer * dim * dim, dim, dim);

            // Attention residual addition
            Kernels.accum(runState.x, runState.xb2, dim);

            // FFN pre-normalization
            Kernels.rmsNorm(runState.xb, runState.x, weights.rmsFfnWeight, layer * dim, dim, config.getLayerNormRMSEpsilon());

            // FNN gate/up projections
            Linear.matmul(runState.hb, runState.xb, weights.w1, layer * dim * config.getHiddenDimensions(), dim, config.getHiddenDimensions());
            Linear.matmul(runState.hb2, runState.xb, weights.w3, layer * dim * config.getHiddenDimensions(), dim, config.getHiddenDimensions());

            // FNN gate activation
            FfnActivation.swiGlu(runState, config);

            // FFN output projection
            Linear.matmul(runState.xb, runState.hb, weights.w2, layer * dim * config.getHiddenDimensions(), config.getHiddenDimensions(), dim);

            // FFN residual addition
            Kernels.accum(runState.x, runState.xb, dim);
        }

        // Final RMS normalization
        Kernels.rmsNorm(runState.x, runState.x, weights.rmsFinalWeight, 0, dim, config.getLayerNormRMSEpsilon());

        // Classifier / Output projection to logits
        Linear.matmul(runState.logits, runState.x, weights.classifier, 0, dim, config.getVocabSize());
    }
}
