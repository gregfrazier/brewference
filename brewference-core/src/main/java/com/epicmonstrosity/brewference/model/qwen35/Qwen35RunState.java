package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.transformer.RunState;
import com.epicmonstrosity.brewference.transformer.attention.AttentionPattern;
import com.epicmonstrosity.brewference.transformer.cache.Fp16KvCache;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetSpec;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetState;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNetWorkspace;

/**
 * Qwen3.5 run state: the inherited activation buffers plus everything the Gated Delta Net needs —
 * the causal depthwise convolution history and the per-head recurrent delta state.
 * <p>
 * Recurrent state lives here rather than in {@link Qwen35Transformer} because the transformer is
 * shared by the runner while sessions come from {@code ModelRunner.createSession()}: mutable state on
 * the transformer would leak between sessions and survive {@code session.reset()}. A fresh instance is
 * built per session by {@code Qwen35ModelRunner.allocateRunState}, and that allocation is what clears
 * the recurrent state.
 * <p>
 * Only full-attention layers get a real KV cache; GDN layers get capacity 1 (the cache implementations
 * reject capacity &le; 0 and GDN layers never touch their cache), and only GDN layers get conv/delta
 * state buffers.
 */
public final class Qwen35RunState extends RunState {
    private static final long DELTA_STATE_BUDGET_BYTES = 1L << 30;

    /** Raw output of the gated {@code attn_q} matmul: per head [headSize query | headSize gate]. */
    public final float[] qGate;
    /** Gate values extracted from {@link #qGate}, same layout as {@code q}. */
    public final float[] attnGate;

    /** Raw {@code attn_qkv} projection (convChannels wide). */
    public final float[] ssmQkv;
    /** {@code attn_gate} projection — the z gate (innerSize wide). */
    public final float[] ssmZ;
    /** Convolution output after the causal depthwise conv (convChannels wide). */
    public final float[] ssmConvOut;
    /** Split conv output: Q, K and V, each innerSize wide (dtRank heads x headVDim). */
    public final float[] ssmQ, ssmK, ssmV;
    /** Per-head decay gate and write strength (dtRank wide). */
    public final float[] ssmGate, ssmBeta;
    /** Normalised per-head output before {@code ssm_out} (innerSize wide). */
    public final float[] ssmOut;
    /** Per-head scratch reused across heads so the delta-rule loop never allocates; head h uses {@code h * headValueDim}. */
    public final float[] ssmSk, ssmD;

    /** Per GDN layer: {@code (convKernel - 1) x convChannels}, row-major {@code k * convChannels + c}. Null on full-attention layers. */
    public final float[][] convState;
    /** Per GDN layer: {@code dtRank x headVDim x headVDim}, layout {@code h * HV^2 + j * HV + i}. Null on full-attention layers. */
    public final float[][] deltaState;
    private final GatedDeltaNetWorkspace gatedDeltaNetWorkspace;
    private final GatedDeltaNetState[] gatedDeltaNetStates;

    public Qwen35RunState(final Config config, final AttentionPattern attentionPattern) {
        final int dim = config.getTransformerDimensions();
        final int headSize = config.getHeadSize();
        final int qDim = config.getQueryAttentionWidth();
        final int qGateWidth = config.getGatedQueryWidth();
        final int kvDim = config.getKeyValueDim();
        final int hiddenDim = config.getHiddenDimensions();
        final int innerSize = config.getInnerSize();
        final int convCh = config.getConvChannels();
        final int headVDim = config.getTimeStepRank() > 0 ? config.getSsmHeadValueDim() : 0;
        final int dtRank = config.getTimeStepRank();
        final int convKernel = config.getConvKernel();
        final int maxSeqLen = config.getMaxSequenceLength();
        final int nLayers = config.getNumLayers();

        if (dim <= 0 || headSize <= 0 || qDim <= 0 || kvDim <= 0 || innerSize <= 0 || convCh <= 0) {
            throw new IllegalStateException("qwen35 run state needs positive dimensions, got dim=%d headSize=%d qDim=%d kvDim=%d innerSize=%d convChannels=%d"
                    .formatted(dim, headSize, qDim, kvDim, innerSize, convCh));
        }

        // AttentionEngine writes attention output into xb at head offsets up to qDim.
        final int xbWidth = Math.max(dim, qDim);

        this.x = new float[dim];
        this.xb = new float[xbWidth];
        this.xb2 = new float[dim];
        this.hb = new float[hiddenDim];
        this.hb2 = new float[hiddenDim];
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[config.getNumHeads() * maxSeqLen];
        this.logits = new float[config.getVocabSize()];
        // Legacy flat FP32 caches stay unused on this path; the per-layer KvCache is what runs.
        this.key_cache = null;
        this.value_cache = null;
        this.kvCache = new Fp16KvCache(kvDim, capacityPerLayer(config, attentionPattern));

        this.qGate = new float[qGateWidth];
        this.attnGate = new float[qDim];
        this.ssmQkv = new float[convCh];
        this.ssmZ = new float[innerSize];
        this.ssmConvOut = new float[convCh];
        this.ssmQ = new float[innerSize];
        this.ssmK = new float[innerSize];
        this.ssmV = new float[innerSize];
        this.ssmGate = new float[dtRank];
        this.ssmBeta = new float[dtRank];
        this.ssmOut = new float[innerSize];
        this.ssmSk = new float[Math.multiplyExact(dtRank, headVDim)];
        this.ssmD = new float[Math.multiplyExact(dtRank, headVDim)];

        this.convState = new float[nLayers][];
        this.deltaState = new float[nLayers][];

        final int gdnLayers = countGdnLayers(config, nLayers);
        if (gdnLayers > 0) {
            guardRecurrentMemory(config, gdnLayers, dtRank, headVDim, convKernel, convCh);
            for (int layer = 0; layer < nLayers; layer++) {
                if (config.isFullAttentionLayer(layer)) {
                    continue;
                }
                convState[layer] = new float[Math.multiplyExact(convKernel - 1, convCh)];
                deltaState[layer] = new float[Math.multiplyExact(dtRank, Math.multiplyExact(headVDim, headVDim))];
            }
        }

        final GatedDeltaNetSpec ssmSpec = Qwen35GatedDeltaNetAdapter.spec(config);
        this.gatedDeltaNetWorkspace = new GatedDeltaNetWorkspace(xb, ssmQkv, ssmZ, ssmConvOut,
                ssmQ, ssmK, ssmV, ssmGate, ssmBeta, ssmOut, ssmSk, ssmD);
        this.gatedDeltaNetStates = new GatedDeltaNetState[nLayers];
        for (int layer = 0; layer < nLayers; layer++) {
            if (convState[layer] != null) {
                gatedDeltaNetStates[layer] = new GatedDeltaNetState(convState[layer], deltaState[layer]);
            }
        }
        gatedDeltaNetWorkspace.validate(ssmSpec);
    }

    public GatedDeltaNetWorkspace gatedDeltaNetWorkspace() {
        return gatedDeltaNetWorkspace;
    }

    public GatedDeltaNetState gatedDeltaNetState(final int layer) {
        if (layer < 0 || layer >= gatedDeltaNetStates.length || gatedDeltaNetStates[layer] == null) {
            throw new IllegalStateException("qwen35 layer %d has no Gated Delta Net state".formatted(layer));
        }
        return gatedDeltaNetStates[layer];
    }

    private static int[] capacityPerLayer(final Config config, final AttentionPattern attentionPattern) {
        final int[] capacity = new int[config.getNumLayers()];
        for (int layer = 0; layer < capacity.length; layer++) {
            capacity[layer] = attentionPattern == null
                    ? config.getMaxSequenceLength()
                    : attentionPattern.cacheCapacityForLayer(layer, config);
        }
        return capacity;
    }

    private static int countGdnLayers(final Config config, final int nLayers) {
        int count = 0;
        for (int layer = 0; layer < nLayers; layer++) {
            if (!config.isFullAttentionLayer(layer)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Fail loudly on an implausible recurrent-state footprint instead of OOMing mid-allocation.
     */
    private static void guardRecurrentMemory(final Config config, final int gdnLayers, final int dtRank,
                                             final int headVDim, final int convKernel, final int convCh) {
        if (dtRank <= 0 || headVDim <= 0 || convKernel <= 1 || convCh <= 0) {
            throw new IllegalStateException("qwen35 SSM metadata is incomplete: time_step_rank=%d head_value_dim=%d conv_kernel=%d conv_channels=%d"
                    .formatted(dtRank, headVDim, convKernel, convCh));
        }
        try {
            final long floatsPerLayer = (long) dtRank * headVDim * headVDim;
            final long bytes = Math.multiplyExact(floatsPerLayer, gdnLayers) * Float.BYTES;
            if (bytes > DELTA_STATE_BUDGET_BYTES) {
                throw new IllegalStateException("qwen35 delta state would need %.1f MB for %d SSM layers (dt_rank=%d, head_value_dim=%d); refusing to allocate"
                        .formatted(bytes / (1024.0 * 1024.0), gdnLayers, dtRank, headVDim));
            }
        } catch (final ArithmeticException e) {
            throw new IllegalStateException("qwen35 delta state size overflows: dt_rank=%d head_value_dim=%d layers=%d"
                    .formatted(dtRank, headVDim, gdnLayers), e);
        }
    }
}
