package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.gguf.Reader;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.tensor.CompositeTensor;
import com.epicmonstrosity.brewference.tensor.CompositeQuantizedTensor;
import com.epicmonstrosity.brewference.tensor.MappedF16Tensor;
import com.epicmonstrosity.brewference.tensor.MappedF32Tensor;
import com.epicmonstrosity.brewference.tensor.QuantizedSegmentTensor;
import com.epicmonstrosity.brewference.tensor.QuantizedTensor;
import com.epicmonstrosity.brewference.tensor.Tensor;
import com.epicmonstrosity.brewference.transformer.rope.LongRopeScaling;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Creates read-only, zero-copy views of supported GGUF tensor payloads.
 * TODO: Refactor
 */
public final class GgufWeightsLoader {
    private final Reader ggufReader;
    private final MemorySegment fileData;

    public GgufWeightsLoader(final Reader ggufReader, final MemorySegment fileData) {
        this.ggufReader = java.util.Objects.requireNonNull(ggufReader, "ggufReader");
        this.fileData = java.util.Objects.requireNonNull(fileData, "fileData");
        if (!fileData.isReadOnly()) {
            throw new IllegalArgumentException("GGUF file memory must be read-only");
        }
    }

    public QuantizedWeights load() throws IOException {
        return load(null);
    }

    /**
     * Builds the same transformer-facing tensor layout as the allocating loader, using only
     * mapped payload views, no-copy slices, and no-copy composite views.
     */
    public QuantizedWeights load(final Config config) throws IOException {
        final QuantizedWeights weights = new QuantizedWeights();
        if (config == null) {
            loadRawTensors(weights);
            tieClassifier(weights);
            return weights;
        }
        final int numLayers = config.getNumLayers();
        final QuantizedTensor[] wq = new QuantizedTensor[numLayers];
        final QuantizedTensor[] wk = new QuantizedTensor[numLayers];
        final QuantizedTensor[] wv = new QuantizedTensor[numLayers];
        final QuantizedTensor[] wo = new QuantizedTensor[numLayers];
        final QuantizedTensor[] w1 = new QuantizedTensor[numLayers];
        final QuantizedTensor[] w2 = new QuantizedTensor[numLayers];
        final QuantizedTensor[] w3 = new QuantizedTensor[numLayers];
        final Tensor[] qBias = new Tensor[numLayers];
        final Tensor[] kBias = new Tensor[numLayers];
        final Tensor[] vBias = new Tensor[numLayers];
        final Tensor[] rmsAtt = new Tensor[numLayers];
        final Tensor[] rmsFfn = new Tensor[numLayers];
        final Tensor[] postAtt = new Tensor[numLayers];
        final Tensor[] postFfn = new Tensor[numLayers];
        final Tensor[] rmsQ = new Tensor[numLayers];
        final Tensor[] rmsK = new Tensor[numLayers];
        final QuantizedTensor[] attnQkv = new QuantizedTensor[numLayers];
        final QuantizedTensor[] attnGate = new QuantizedTensor[numLayers];
        final QuantizedTensor[] ssmAlpha = new QuantizedTensor[numLayers];
        final QuantizedTensor[] ssmBeta = new QuantizedTensor[numLayers];
        final QuantizedTensor[] ssmOut = new QuantizedTensor[numLayers];
        final Tensor[] ssmA = new Tensor[numLayers];
        final Tensor[] ssmConv1d = new Tensor[numLayers];
        final Tensor[] ssmDtBias = new Tensor[numLayers];
        final Tensor[] ssmNorm = new Tensor[numLayers];
        final Pattern blockName = Pattern.compile("^blk\\.(\\d+)\\.(.+)$");

        for (final Reader.TensorInfo tensor : ggufReader.getTensors()) {
            final long elements = elementCount(tensor);
            final Tensor view = createView(tensor, elements);
            weights.putTensor(tensor.name, view);

            assignDirectTensor(weights, tensor.name, view);

            final Matcher matcher = blockName.matcher(tensor.name);
            if (!matcher.matches())
                continue;
            final int layer = Integer.parseInt(matcher.group(1));
            if (layer < 0 || layer >= numLayers)
                throw new IOException("Tensor has out-of-range layer index: " + tensor.name);

            switch (matcher.group(2)) {
                case "attn_q.weight" -> wq[layer] = quantizedSlot(tensor.name, view);
                case "attn_k.weight" -> wk[layer] = quantizedSlot(tensor.name, view);
                case "attn_v.weight" -> wv[layer] = quantizedSlot(tensor.name, view);
                case "attn_qkv.weight" -> {
                    if (config.isHybridAttention()) {
                        // Hybrid arches (Qwen3.5): on SSM layers this is a single fused projection
                        // dim -> convChannels, not a q/k/v triple. Splitting it would be wrong.
                        attnQkv[layer] = quantizedSlot(tensor.name, view);
                    } else {
                        final int inDim = fusedInputDim(tensor, config);
                        final var split = GgufTensorSplitter.splitFusedQkv(quantizedSlot(tensor.name, view),
                                Math.multiplyExact(config.getQueryAttentionWidth(), inDim),
                                Math.multiplyExact(config.getKeyValueDim(), inDim),
                                Math.multiplyExact(config.getKeyValueDim(), inDim), tensor.name);
                        wq[layer] = split.q(); wk[layer] = split.k(); wv[layer] = split.v();
                    }
                }
                case "attn_q.bias" -> qBias[layer] = elementWise(tensor.name, view);
                case "attn_k.bias" -> kBias[layer] = elementWise(tensor.name, view);
                case "attn_v.bias" -> vBias[layer] = elementWise(tensor.name, view);
                case "attn_qkv.bias" -> {
                    // Hybrid arches have no qkv bias in practice; if one appears it stays in the raw
                    // tensor map rather than being split into q/k/v biases.
                    if (!config.isHybridAttention()) {
                        final var split = GgufTensorSplitter.splitFusedQkv(elementWise(tensor.name, view),
                                config.getQueryAttentionWidth(), config.getKeyValueDim(), config.getKeyValueDim(), tensor.name);
                        qBias[layer] = split.q(); kBias[layer] = split.k(); vBias[layer] = split.v();
                    }
                }
                case "attn_output.weight" -> wo[layer] = quantizedSlot(tensor.name, view);
                case "ffn_gate.weight" -> w1[layer] = quantizedSlot(tensor.name, view);
                case "ffn_down.weight" -> w2[layer] = quantizedSlot(tensor.name, view);
                case "ffn_up.weight" -> {
                    final QuantizedTensor up = quantizedSlot(tensor.name, view);
                    final int projectionElements = Math.multiplyExact(config.getTransformerDimensions(), config.getHiddenDimensions());
                    if (elements == Math.multiplyExact((long) projectionElements, 2)) {
                        final var split = GgufTensorSplitter.splitFusedGateUp(up, projectionElements, tensor.name);
                        w1[layer] = split.gate(); w3[layer] = split.up();
                    } else w3[layer] = up;
                }
                case "attn_norm.weight" -> rmsAtt[layer] = elementWise(tensor.name, view);
                case "ffn_norm.weight" -> rmsFfn[layer] = elementWise(tensor.name, view);
                case "post_attention_norm.weight" -> postAtt[layer] = elementWise(tensor.name, view);
                case "post_ffw_norm.weight" -> postFfn[layer] = elementWise(tensor.name, view);
                case "attn_q_norm.weight" -> rmsQ[layer] = elementWise(tensor.name, view);
                case "attn_k_norm.weight" -> rmsK[layer] = elementWise(tensor.name, view);

                // Gated Delta Net (Qwen3.5 linear-attention blocks)
                case "attn_gate.weight" -> attnGate[layer] = quantizedSlot(tensor.name, view);
                case "ssm_a" -> ssmA[layer] = elementWise(tensor.name, view);
                case "ssm_alpha.weight" -> ssmAlpha[layer] = quantizedSlot(tensor.name, view);
                case "ssm_beta.weight" -> ssmBeta[layer] = quantizedSlot(tensor.name, view);
                case "ssm_conv1d.weight" -> ssmConv1d[layer] = elementWise(tensor.name, view);
                case "ssm_dt.bias" -> ssmDtBias[layer] = elementWise(tensor.name, view);
                case "ssm_norm.weight" -> ssmNorm[layer] = elementWise(tensor.name, view);
                case "ssm_out.weight" -> ssmOut[layer] = quantizedSlot(tensor.name, view);

                default -> { }
            }
        }
        weights.wq = concatenateQuantized(wq);
        weights.wk = concatenateQuantized(wk);
        weights.wv = concatenateQuantized(wv);
        weights.wo = concatenateQuantized(wo);
        weights.w1 = concatenateQuantized(w1);
        weights.w2 = concatenateQuantized(w2);
        weights.w3 = concatenateQuantized(w3);

        weights.qBias = concatenateTensor(qBias);
        weights.kBias = concatenateTensor(kBias);
        weights.vBias = concatenateTensor(vBias);

        weights.rmsAttWeight = concatenateTensor(rmsAtt);
        weights.rmsFfnWeight = concatenateTensor(rmsFfn);

        weights.postAttWeight = concatenateTensor(postAtt);
        weights.postFfnWeight = concatenateTensor(postFfn);

        weights.rmsQWeight = concatenateTensor(rmsQ);
        weights.rmsKWeight = concatenateTensor(rmsK);

        // Per-layer views of the very same tensors. Required by architectures whose layers are not
        // shape-uniform; null when the GGUF carried none of that tensor family.
        weights.wqLayer = keep(wq);
        weights.wkLayer = keep(wk);
        weights.wvLayer = keep(wv);
        weights.woLayer = keep(wo);
        weights.w1Layer = keep(w1);
        weights.w2Layer = keep(w2);
        weights.w3Layer = keep(w3);

        weights.qBiasLayer = keep(qBias);
        weights.kBiasLayer = keep(kBias);
        weights.vBiasLayer = keep(vBias);

        weights.rmsAttLayer = keep(rmsAtt);
        weights.rmsFfnLayer = keep(rmsFfn);
        weights.postAttLayer = keep(postAtt);
        weights.postFfnLayer = keep(postFfn);
        weights.rmsQLayer = keep(rmsQ);
        weights.rmsKLayer = keep(rmsK);

        weights.attnQkvLayer = keep(attnQkv);
        weights.attnGateLayer = keep(attnGate);
        weights.ssmAlphaLayer = keep(ssmAlpha);
        weights.ssmBetaLayer = keep(ssmBeta);
        weights.ssmOutLayer = keep(ssmOut);
        weights.ssmALayer = keep(ssmA);
        weights.ssmConv1dLayer = keep(ssmConv1d);
        weights.ssmDtBiasLayer = keep(ssmDtBias);
        weights.ssmNormLayer = keep(ssmNorm);

        tieClassifier(weights);
        if (weights.ropeFactorsShort != null && weights.ropeFactorsLong != null) {
            weights.ropeScaling = new LongRopeScaling(weights.ropeFactorsShort, weights.ropeFactorsLong,
                    config.getRopeDimCount(), config.getRopeFrequencyBase(), config.getRopeScalingOriginalContextLength(),
                    config.getRopeScalingAttnFactor());
        }

        return weights;
    }

    private void loadRawTensors(final QuantizedWeights weights) throws IOException {
        for (final Reader.TensorInfo tensor : ggufReader.getTensors()) {
            final Tensor view = createView(tensor, elementCount(tensor));
            weights.putTensor(tensor.name, view);
            assignDirectTensor(weights, tensor.name, view);
        }
    }

    private static void tieClassifier(final QuantizedWeights weights) {
        if (weights.classifier == null)
            weights.classifier = weights.tokenEmbeddingTable;
    }

    private static int fusedInputDim(final Reader.TensorInfo tensor, final Config config) {
        return tensor.dims.length >= 2 ? Math.toIntExact(tensor.dims[0]) : config.getTransformerDimensions();
    }

    private static QuantizedTensor concatenateQuantized(final QuantizedTensor[] parts) {
        if (Arrays.stream(parts).allMatch(java.util.Objects::isNull)) return null;
        final QuantizedTensor[] nonNullParts = Arrays.stream(parts).filter(java.util.Objects::nonNull).toArray(QuantizedTensor[]::new);
        return new CompositeQuantizedTensor(nonNullParts);
    }

    private static Tensor concatenateTensor(final Tensor[] parts) {
        if (Arrays.stream(parts).allMatch(java.util.Objects::isNull)) return null;
        return new CompositeTensor(parts);
    }

    private static QuantizedTensor[] keep(final QuantizedTensor[] parts) {
        return Arrays.stream(parts).allMatch(java.util.Objects::isNull) ? null : parts;
    }

    private static Tensor[] keep(final Tensor[] parts) {
        return Arrays.stream(parts).allMatch(java.util.Objects::isNull) ? null : parts;
    }

    /**
     * Creates a view of a tensor from the GGUF file data based on the provided tensor metadata and element count.
     */
    private Tensor createView(final Reader.TensorInfo tensor, final long elements) throws IOException {
        final GgmlType ggmlType = tensor.getGgmlType();
        final long byteCount;
        try {
            byteCount = switch (ggmlType) {
                case F32 -> Math.multiplyExact(elements, (long) Float.BYTES);
                case F16 -> Math.multiplyExact(elements, (long) Short.BYTES);
                case BF16 -> Math.multiplyExact(elements, (long) Short.BYTES);
                default -> {
                    if (elements % ggmlType.getBlockSize() != 0) {
                        throw new IOException(ggmlType.name() + " tensor element count is not divisible by "
                                + ggmlType.getBlockSize() + ": " + tensor.name);
                    }
                    yield ggmlType.byteSizeFor(elements);
                }
            };
        } catch (final ArithmeticException e) {
            throw new IOException("Tensor byte size overflows: " + tensor.name, e);
        } catch (final RuntimeException e) {
            throw new IOException("Unsupported GGUF tensor type %d for %s".formatted(tensor.type, tensor.name), e);
        }

        final long offset;
        try {
            offset = Math.addExact(ggufReader.getDataOffset(), tensor.offset);
        } catch (final ArithmeticException e) {
            throw new IOException("Tensor offset overflows: " + tensor.name, e);
        }
        if (offset < 0 || byteCount > fileData.byteSize() - offset) {
            throw new IOException("Tensor payload is outside mapped GGUF data: " + tensor.name);
        }

        final MemorySegment data = fileData.asSlice(offset, byteCount);
        return switch (ggmlType) {
            case F32 -> new MappedF32Tensor(data, elements);
            case F16 -> new MappedF16Tensor(data, elements);
            // BF16 is not block-quantized but is not a FloatTensor either: QuantizedSegmentTensor
            // already reads it (bits << 16) and QuantizedKernels registers dotBF16, so the same
            // view type serves both matmul and element-wise slots.
            case BF16 -> new QuantizedSegmentTensor(data, GgmlType.BF16, elements);
            default -> {
                if (ggmlType.isQuantized()) {
                    yield new QuantizedSegmentTensor(data, ggmlType, elements);
                }
                throw new IOException("Unhandled tensor type: %s for %s".formatted(ggmlType, tensor.name));
            }
        };
    }

    private static long elementCount(final Reader.TensorInfo tensor) throws IOException {
        long elements = 1;
        for (final long dimension : tensor.dims) {
            if (dimension <= 0) {
                throw new IOException("Bad tensor dimension in %s: %d".formatted(tensor.name, dimension));
            }
            try {
                elements = Math.multiplyExact(elements, dimension);
            } catch (final ArithmeticException e) {
                throw new IOException("Tensor element count overflows: " + tensor.name, e);
            }
        }
        return elements;
    }

    private static void assignDirectTensor(final QuantizedWeights weights, final String name,
                                           final Tensor tensor) {
        switch (name) {
            case "token_embd.weight" -> weights.tokenEmbeddingTable = quantizedSlot(name, tensor);
            case "output.weight" -> weights.classifier = quantizedSlot(name, tensor);
            case "output_norm.weight" -> weights.rmsFinalWeight = elementWise(name, tensor);
            case "rope_factors_long.weight" -> weights.ropeFactorsLong = elementWise(name, tensor);
            case "rope_factors_short.weight" -> weights.ropeFactorsShort = elementWise(name, tensor);
            default -> { }
        }
    }

    /**
     * Accepts a matmul weight slot that a checkpoint may store either block-quantized or as plain
     * F32/F16 floats.
     * <p>
     * Conversions disagree about which tensors are worth quantizing: some exports keep norms, biases
     * and the whole gated-delta-net family ({@code ssm_alpha.weight}, {@code ssm_beta.weight},
     * {@code attn_gate.weight}, the hybrid {@code attn_qkv.weight}) in F32 while quantizing the
     * attention and FFN bulk; other exports keep the attention and FFN bulk in F32. A mapped
     * F32/F16 payload is re-expressed as a {@link QuantizedSegmentTensor} over the very same memory
     * - no copy, no requantization, no numerics change - because {@code QuantizedKernels} already
     * registers F32 and F16 dot kernels, so the matmul stays on its vectorized segment path instead
     * of falling back to per-element {@code value()} reads. A block-quantized tensor is returned
     * unchanged, so quantized checkpoints stay bit-for-bit identical.
     */
    static QuantizedTensor quantizedSlot(final String name, final Tensor tensor) {
        if (tensor instanceof final QuantizedTensor q)
            return q;
        if (tensor instanceof final MappedF32Tensor f32)
            return new QuantizedSegmentTensor(f32.data(), GgmlType.F32, f32.elementCount());
        if (tensor instanceof final MappedF16Tensor f16)
            return new QuantizedSegmentTensor(f16.data(), GgmlType.F16, f16.elementCount());
        throw new IllegalArgumentException("Expected a quantized or mapped F32/F16 tensor for %s, got %s"
                .formatted(name, tensor.getClass().getSimpleName()));
    }

    /**
     * Accepts an element-wise slot (bias, norm, rope factor, ssm_a / ssm_conv1d / ssm_dt / ssm_norm).
     * <p>
     * Any GGML type is accepted: the tensor is stored as-is and a quantized payload dequantizes
     * lazily through {@link Tensor#value(long)} when a kernel reads it. No cast, no copy, no eager
     * dequantization.
     */
    static Tensor elementWise(final String name, final Tensor tensor) {
        if (tensor != null)
            return tensor;
        throw new IllegalArgumentException("Missing tensor for %s".formatted(name));
    }
}
