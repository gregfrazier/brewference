package com.epicmonstrosity.brewference.model.qwen35;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.gguf.loader.GgufConfigParser;
import com.epicmonstrosity.brewference.tensor.FloatArrayTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.transformer.attention.LayerContext;
import com.epicmonstrosity.brewference.transformer.rope.PartialRope;
import com.epicmonstrosity.brewference.transformer.ssm.GatedDeltaNet;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Numerics for the Qwen3.5 pieces: Gated Delta Net, gated full attention, partial RoPE, and the
 * GGUF metadata the hybrid graph depends on.
 */
class Qwen35NumericsTest {
    private static final float EPS = 1e-6f;

    private static QuantizedTensor f32(final float... values) {
        return new QuantizedSegmentTensor(MemorySegment.ofArray(values).asReadOnly(), GgmlType.F32, values.length);
    }

    private static FloatArrayTensor floats(final float... values) {
        return new FloatArrayTensor(values);
    }

    // ---------------------------------------------------------------------------------------
    // Config
    // ---------------------------------------------------------------------------------------

    @Test
    void readsQwen35SsmAndKeyLengthMetadata() {
        final Config config = GgufConfigParser.parseCommon(Map.ofEntries(
                Map.entry("general.architecture", "qwen35"),
                Map.entry("qwen35.embedding_length", 2560),
                Map.entry("qwen35.attention.head_count", 32),
                Map.entry("qwen35.attention.head_count_kv", 2),
                Map.entry("qwen35.attention.key_length", 256),
                Map.entry("qwen35.full_attention_interval", 4),
                Map.entry("qwen35.ssm.conv_kernel", 4),
                Map.entry("qwen35.ssm.inner_size", 4096),
                Map.entry("qwen35.ssm.time_step_rank", 16),
                Map.entry("qwen35.ssm.group_count", 16),
                Map.entry("qwen35.ssm.state_size", 256)
        ));

        assertEquals(256, config.getAttentionKeyLength());
        assertEquals(4, config.getFullAttentionInterval());
        assertEquals(4, config.getConvKernel());
        assertEquals(4096, config.getInnerSize());
        assertEquals(16, config.getTimeStepRank());
        assertEquals(16, config.getGroupCount());
        assertEquals(256, config.getStateSize());

        // attn_qkv is inner_size value channels plus 2 * group_count * state_size query/key channels.
        assertEquals(4096 + 2 * 16 * 256, config.getConvChannels());
        assertEquals(4096 / 16, config.getSsmHeadValueDim());
        assertEquals(2 * config.getQueryAttentionWidth(), config.getGatedQueryWidth());
        assertTrue(config.isHybridAttention());

        // interval 4: (layer + 1) % 4 == 0 -> layers 3,7,... are full attention, the rest are SSM.
        assertFalse(config.isFullAttentionLayer(0));
        assertFalse(config.isFullAttentionLayer(2));
        assertTrue(config.isFullAttentionLayer(3));
        assertTrue(config.isFullAttentionLayer(7));
    }

    @Test
    void absentFullAttentionIntervalTreatsEveryLayerAsFullAttention() {
        final Config config = new Config().setFullAttentionInterval(0);
        assertTrue(config.isFullAttentionLayer(0));
        assertTrue(config.isFullAttentionLayer(7));
    }

    // ---------------------------------------------------------------------------------------
    // Partial RoPE
    // ---------------------------------------------------------------------------------------

    @Test
    void partialRopeRotatesOnlyTheFirstDimensionCountSplitHalfPaired() {
        final float[] query = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

        // headSize 8, rope.dimension_count 4 -> pairs (0,2) and (1,3); theta_j = base^(-2j/4).
        PartialRope.apply(query, 1, 8, 4, 1_000_000.0, 1);

        final double cos0 = Math.cos(1.0);
        final double sin0 = Math.sin(1.0);
        final double theta1 = Math.pow(1_000_000.0, -2.0 / 4.0); // 1e-3
        final double cos1 = Math.cos(theta1);
        final double sin1 = Math.sin(theta1);

        assertEquals(1.0 * cos0 - 3.0 * sin0, query[0], 1e-5);
        assertEquals(3.0 * cos0 + 1.0 * sin0, query[2], 1e-5);
        assertEquals(2.0 * cos1 - 4.0 * sin1, query[1], 1e-5);
        assertEquals(4.0 * cos1 + 2.0 * sin1, query[3], 1e-5);

        // Everything past rope.dimension_count passes through untouched.
        assertEquals(5.0f, query[4], 0.0f);
        assertEquals(6.0f, query[5], 0.0f);
        assertEquals(7.0f, query[6], 0.0f);
        assertEquals(8.0f, query[7], 0.0f);
    }

    @Test
    void partialRopeFallsBackToWholeHeadWhenDimensionCountAbsent() {
        final float[] query = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        PartialRope.apply(query, 1, 8, 0, 1_000_000.0, 1);

        // Whole head: pairs (0,4),(1,5),(2,6),(3,7) with theta_0 = 1.
        assertEquals(Math.cos(1.0), query[0], 1e-5);
        assertEquals(Math.sin(1.0), query[4], 1e-5);
        for (int i = 1; i < 4; i++) {
            assertEquals(0.0f, query[i], 1e-5f);
        }
    }

    @Test
    void partialRopeAtPositionZeroIsIdentity() {
        final float[] query = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        final float[] expected = query.clone();

        PartialRope.apply(query, 2, 4, 4, 1_000_000.0, 0);

        assertArrayEquals(expected, query, 0.0f);
    }

    // ---------------------------------------------------------------------------------------
    // Activation clamps
    // ---------------------------------------------------------------------------------------

    @Test
    void softplusAndSigmoidBoundaries() {
        assertEquals(25.0f, GatedDeltaNet.softplus(25.0f), 0.0f);
        assertEquals((float) Math.exp(-25.0), GatedDeltaNet.softplus(-25.0f), 1e-12f);
        assertEquals((float) Math.log1p(Math.exp(3.0)), GatedDeltaNet.softplus(3.0f), 1e-6f);

        assertEquals(0.5f, GatedDeltaNet.sigmoid(0.0f), 0.0f);
        assertEquals(1.0f, GatedDeltaNet.sigmoid(80.0f), 0.0f);
        assertEquals((float) (1.0 / (1.0 + Math.exp(80.0))), GatedDeltaNet.sigmoid(-80.0f), 1e-40f);
        assertEquals(0.0f, GatedDeltaNet.silu(0.0f), 0.0f);
        assertEquals(80.0f, GatedDeltaNet.silu(80.0f), 1e-6f);
    }

    // ---------------------------------------------------------------------------------------
    // Gated Delta Net
    // ---------------------------------------------------------------------------------------

    /**
     * A tiny SSM layer: dim 1, kernel-2 conv whose first tap is zero (so every step sees the same
     * conv output), and enough metadata for {@link GatedDeltaNet#validate} to accept it.
     */
    private static Config ssmConfig(final int innerSize, final int timeStepRank,
                                    final int groupCount, final int stateSize) {
        return new Config()
                .setArchitecture("qwen35")
                .setTransformerDimensions(1)
                .setNumHeads(1)
                .setNumKVHeads(1)
                .setHeadSize(1)
                .setHiddenDimensions(1)
                .setVocabSize(1)
                .setNumLayers(3)
                .setMaxSequenceLength(4)
                // interval 3 -> layers 2,5 are full attention, so layer 1 is a GDN layer.
                .setFullAttentionInterval(3)
                .setInnerSize(innerSize)
                .setTimeStepRank(timeStepRank)
                .setGroupCount(groupCount)
                .setStateSize(stateSize)
                .setConvKernel(2)
                .setLayerNormRMSEpsilon(EPS);
    }

    private static QuantizedWeights ssmWeights(final Config config, final float[] qkv, final float[] gate,
                                               final float[] alpha, final float[] beta, final float[] out,
                                               final float[] dtBias, final float[] a, final float[] norm) {
        final float[] conv = new float[config.getConvChannels() * config.getConvKernel()];
        Arrays.fill(conv, 0.0f);
        for (int channel = 0; channel < config.getConvChannels(); channel++) {
            // [tap0, tap1] per channel: only the current token contributes.
            conv[channel * config.getConvKernel() + 1] = 1.0f;
        }

        final QuantizedWeights weights = new QuantizedWeights();
        weights.attnQkvLayer = new QuantizedTensor[]{null, f32(qkv)};
        weights.attnGateLayer = new QuantizedTensor[]{null, f32(gate)};
        weights.ssmAlphaLayer = new QuantizedTensor[]{null, f32(alpha)};
        weights.ssmBetaLayer = new QuantizedTensor[]{null, f32(beta)};
        weights.ssmOutLayer = new QuantizedTensor[]{null, f32(out)};
        weights.ssmALayer = new FloatArrayTensor[]{null, floats(a)};
        weights.ssmConv1dLayer = new FloatArrayTensor[]{null, floats(conv)};
        weights.ssmDtBiasLayer = new FloatArrayTensor[]{null, floats(dtBias)};
        weights.ssmNormLayer = new FloatArrayTensor[]{null, floats(norm)};
        return weights;
    }

    private static float runSsm(final Config config, final QuantizedWeights weights, final Qwen35RunState state) {
        state.xb[0] = 1.0f;
        GatedDeltaNet.forward(state.gatedDeltaNetWorkspace(), state.gatedDeltaNetState(1),
                Qwen35GatedDeltaNetAdapter.weights(weights, 1), Qwen35GatedDeltaNetAdapter.spec(config));
        return state.xb[0];
    }

    @Test
    void gatedDeltaNetSingleStepMatchesHandComputedReference() {
        // One head of value dim 1: q and k are unit scalars, so the state is just v * beta.
        final Config config = ssmConfig(1, 1, 1, 1);
        final QuantizedWeights weights = ssmWeights(config,
                new float[]{2.0f, 3.0f, 4.0f},   // attn_qkv: q=2, k=3, v=4 (one channel each)
                new float[]{1.0f},               // attn_gate (z)
                new float[]{0.0f},               // ssm_alpha
                new float[]{1.0f},               // ssm_beta
                new float[]{1.0f},               // ssm_out
                new float[]{0.0f},               // ssm_dt.bias
                new float[]{-0.5f},              // ssm_a
                new float[]{1.0f});              // ssm_norm

        final Qwen35RunState state = new Qwen35RunState(config, new Qwen35AttentionPattern(config));

        final float output = runSsm(config, weights, state);

        // conv+SiLU -> q=1, k=1, v=silu(4)=3.9280587; beta=sigmoid(1)=0.7310586
        // S = k * (v - 0) * beta = 2.8716416
        assertEquals(2.8716416f, state.deltaState[1][0], 1e-5f);

        // The head output is RMS-normalised (value dim 1 -> 1.0), gated by silu(z)=silu(1), then ssm_out.
        assertEquals(GatedDeltaNet.silu(1.0f), output, 1e-5f);
    }

    @Test
    void gatedDeltaNetSecondStepDecaysPriorStateBeforeAddingTheNewDelta() {
        final Config config = ssmConfig(1, 1, 1, 1);
        final QuantizedWeights weights = ssmWeights(config,
                new float[]{2.0f, 3.0f, 4.0f},
                new float[]{1.0f},
                new float[]{0.0f},
                new float[]{1.0f},
                new float[]{1.0f},
                new float[]{0.0f},
                new float[]{-0.5f},
                new float[]{1.0f});

        final Qwen35RunState state = new Qwen35RunState(config, new Qwen35AttentionPattern(config));

        runSsm(config, weights, state);
        final double priorState = state.deltaState[1][0];

        runSsm(config, weights, state);

        // Read this step's own q/k/v back from the kernel buffers.
        final double k = state.ssmK[0];
        final double v = state.ssmV[0];
        final double decay = Math.exp(-0.5 * GatedDeltaNet.softplus(0.0f)); // exp(ssm_a * softplus(dt + dt_bias))
        final double beta = GatedDeltaNet.sigmoid(1.0f);

        final double expectedState = priorState * decay + (v - priorState * decay * k) * beta * k;
        assertEquals(expectedState, state.deltaState[1][0], 1e-5);
        assertTrue(priorState * decay < priorState, "the decay factor must shrink the carried state before the delta is added");
    }

    @Test
    void gatedDeltaNetTilesGroupHeadsAcrossTimeStepHeads() {
        // inner_size 4 / time_step_rank 2 -> two heads of value dim 2.
        // group_count 1 shares one group head; group_count 2 gives each head its own q/k channels.
        final Config oneGroup = ssmConfig(4, 2, 1, 2);
        final Config twoGroups = ssmConfig(4, 2, 2, 2);

        // conv split is [Q(groupCount*stateSize) | K(groupCount*stateSize) | V(innerSize)].
        final float[] oneGroupQkv = {1, 1, 1, 1, 3, 0, 0, 4};
        final float[] twoGroupsQkv = {1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 4};

        // Head 0's value channels get [silu(3), 0]; head 1's get [0, silu(4)].
        final double expected = Math.sqrt(2.0) * GatedDeltaNet.silu(1.0f);

        for (final boolean sharedGroupHead : new boolean[]{true, false}) {
            final Config config = sharedGroupHead ? oneGroup : twoGroups;
            final float[] qkv = sharedGroupHead ? oneGroupQkv : twoGroupsQkv;

            assertEquals(expected, ssmComponent(config, qkv, 0), 1e-5,
                    "head 0 value channel 0");
            assertEquals(0.0f, ssmComponent(config, qkv, 1), 1e-5,
                    "head 0 value channel 1 has no value written to it");
            assertEquals(0.0f, ssmComponent(config, qkv, 2), 1e-5,
                    "head 1 value channel 0 has no value written to it");
            assertEquals(expected, ssmComponent(config, qkv, 3), 1e-5,
                    "head 1 value channel 1");
        }
    }

    @Test
    void gatedDeltaNetRejectsNegativeLayerBeforeResourceLookup() {
        final Config config = ssmConfig(1, 1, 1, 1);

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Qwen35GatedDeltaNetAdapter.weights(new QuantizedWeights(), -1));

        assertTrue(error.getMessage().contains("non-negative"));
    }

    @Test
    void gatedDeltaNetReportsUndersizedLayerWeight() {
        final Config config = ssmConfig(1, 1, 1, 1);
        final QuantizedWeights weights = ssmWeights(config,
                new float[]{2.0f, 3.0f}, // Missing one attn_qkv row element.
                new float[]{1.0f},
                new float[]{0.0f},
                new float[]{1.0f},
                new float[]{1.0f},
                new float[]{0.0f},
                new float[]{-0.5f},
                new float[]{1.0f});
        final Qwen35RunState state = new Qwen35RunState(config, new Qwen35AttentionPattern(config));

        final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> GatedDeltaNet.forward(state.gatedDeltaNetWorkspace(), state.gatedDeltaNetState(1),
                        Qwen35GatedDeltaNetAdapter.weights(weights, 1), Qwen35GatedDeltaNetAdapter.spec(config)));

        assertTrue(error.getMessage().contains("qkv"));
    }

    /** Run the SSM layer with an ssm_out that selects one of the four head/value components. */
    private static float ssmComponent(final Config config, final float[] qkv, final int component) {
        final float[] out = new float[config.getInnerSize()];
        out[component] = 1.0f;

        final QuantizedWeights weights = ssmWeights(config, qkv,
                new float[]{1, 1, 1, 1},   // attn_gate: z = 1 for every head
                new float[]{0, 0},         // ssm_alpha
                new float[]{1, 1},         // ssm_beta
                out,
                new float[]{0, 0},         // ssm_dt.bias
                new float[]{-0.5f, -0.5f}, // ssm_a
                new float[]{1, 1});        // ssm_norm

        final Qwen35RunState state = new Qwen35RunState(config, new Qwen35AttentionPattern(config));
        return runSsm(config, weights, state);
    }

    // ---------------------------------------------------------------------------------------
    // Gated full attention
    // ---------------------------------------------------------------------------------------

    private static Config attentionConfig() {
        return new Config()
                .setArchitecture("qwen35")
                .setTransformerDimensions(4)
                .setNumHeads(1)
                .setNumKVHeads(1)
                .setHeadSize(4)
                .setHiddenDimensions(4)
                .setVocabSize(4)
                .setNumLayers(1)
                .setMaxSequenceLength(8)
                .setFullAttentionInterval(1)
                .setRopeDimCount(0)
                .setRopeFrequencyBase(1_000_000.0f)
                .setLayerNormRMSEpsilon(EPS)
                .setInnerSize(4)
                .setTimeStepRank(1)
                .setGroupCount(1)
                .setStateSize(4)
                .setConvKernel(2);
    }

    private static QuantizedWeights attentionWeights() {
        final QuantizedWeights weights = new QuantizedWeights();
        // attn_q is output-major: rows [0..3] are the head's query, rows [4..7] its gate.
        weights.wqLayer = new QuantizedTensor[]{f32(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                1, 1, 1, 1,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0)};
        weights.wkLayer = new QuantizedTensor[]{f32(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)};
        weights.wvLayer = new QuantizedTensor[]{f32(
                1, 0, 0, 0,
                0, 2, 0, 0,
                0, 0, 3, 0,
                0, 0, 0, 4)};
        weights.woLayer = new QuantizedTensor[]{f32(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)};
        weights.rmsQLayer = new FloatArrayTensor[]{floats(1, 1, 1, 1)};
        weights.rmsKLayer = new FloatArrayTensor[]{floats(1, 1, 1, 1)};
        return weights;
    }

    @Test
    void fullAttentionSplitsTheGatedQueryAndScalesOutputBySigmoidGate() {
        final Config config = attentionConfig();
        final Qwen35AttentionPattern pattern = new Qwen35AttentionPattern(config);
        final Qwen35Transformer transformer = new Qwen35Transformer(config, pattern);
        final Qwen35RunState state = new Qwen35RunState(config, pattern);
        final QuantizedWeights weights = attentionWeights();

        Arrays.fill(state.xb, 1.0f);

        transformer.fullAttention(0, new LayerContext(0, 0, 0, 1), state, weights);

        assertArrayEquals(new float[]{1, 1, 1, 1}, state.q, 1e-4f);
        assertArrayEquals(new float[]{4, 0, 0, 0}, state.attnGate, 1e-5f);

        // Single position, so the attention output == v == [1,2,3,4]; gates are sigmoid([4,0,0,0]).
        assertArrayEquals(new float[]{
                1.0f / (1.0f + (float) Math.exp(-4.0)),
                1.0f,
                1.5f,
                2.0f}, state.x, 1e-5f);
    }

    @Test
    void attentionScalingUsesHeadSizeNotQueryWidth() {
        final Config config = attentionConfig().setHeadSize(256);
        assertEquals((float) Math.sqrt(256.0), new Qwen35AttentionPattern(config).getAttentionScaling(config), 1e-6f);
    }
}
