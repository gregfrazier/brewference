package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.gguf.GgmlType;
import com.epicmonstrosity.brewference.gguf.Reader;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;
import com.epicmonstrosity.brewference.tensor.CompositeFloatTensor;
import com.epicmonstrosity.brewference.tensor.CompositeQuantizedTensor;
import com.epicmonstrosity.brewference.tensor.FloatTensor;
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
        final FloatTensor[] qBias = new FloatTensor[numLayers];
        final FloatTensor[] kBias = new FloatTensor[numLayers];
        final FloatTensor[] vBias = new FloatTensor[numLayers];
        final FloatTensor[] rmsAtt = new FloatTensor[numLayers];
        final FloatTensor[] rmsFfn = new FloatTensor[numLayers];
        final FloatTensor[] postAtt = new FloatTensor[numLayers];
        final FloatTensor[] postFfn = new FloatTensor[numLayers];
        final FloatTensor[] rmsQ = new FloatTensor[numLayers];
        final FloatTensor[] rmsK = new FloatTensor[numLayers];
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
                case "attn_q.weight" -> wq[layer] = quantized(tensor.name, view);
                case "attn_k.weight" -> wk[layer] = quantized(tensor.name, view);
                case "attn_v.weight" -> wv[layer] = quantized(tensor.name, view);
                case "attn_qkv.weight" -> {
                    final int inDim = fusedInputDim(tensor, config);
                    final var split = GgufTensorSplitter.splitFusedQkv(quantized(tensor.name, view),
                            Math.multiplyExact(config.getQueryAttentionWidth(), inDim),
                            Math.multiplyExact(config.getKeyValueDim(), inDim),
                            Math.multiplyExact(config.getKeyValueDim(), inDim), tensor.name);
                    wq[layer] = split.q(); wk[layer] = split.k(); wv[layer] = split.v();
                }
                case "attn_q.bias" -> qBias[layer] = floating(tensor.name, view);
                case "attn_k.bias" -> kBias[layer] = floating(tensor.name, view);
                case "attn_v.bias" -> vBias[layer] = floating(tensor.name, view);
                case "attn_qkv.bias" -> {
                    final var split = GgufTensorSplitter.splitFusedQkv(floating(tensor.name, view),
                            config.getQueryAttentionWidth(), config.getKeyValueDim(), config.getKeyValueDim(), tensor.name);
                    qBias[layer] = split.q(); kBias[layer] = split.k(); vBias[layer] = split.v();
                }
                case "attn_output.weight" -> wo[layer] = quantized(tensor.name, view);
                case "ffn_gate.weight" -> w1[layer] = quantized(tensor.name, view);
                case "ffn_down.weight" -> w2[layer] = quantized(tensor.name, view);
                case "ffn_up.weight" -> {
                    final QuantizedTensor up = quantized(tensor.name, view);
                    final int projectionElements = Math.multiplyExact(config.getTransformerDimensions(), config.getHiddenDimensions());
                    if (elements == Math.multiplyExact((long) projectionElements, 2)) {
                        final var split = GgufTensorSplitter.splitFusedGateUp(up, projectionElements, tensor.name);
                        w1[layer] = split.gate(); w3[layer] = split.up();
                    } else w3[layer] = up;
                }
                case "attn_norm.weight" -> rmsAtt[layer] = floating(tensor.name, view);
                case "ffn_norm.weight" -> rmsFfn[layer] = floating(tensor.name, view);
                case "post_attention_norm.weight" -> postAtt[layer] = floating(tensor.name, view);
                case "post_ffw_norm.weight" -> postFfn[layer] = floating(tensor.name, view);
                case "attn_q_norm.weight" -> rmsQ[layer] = floating(tensor.name, view);
                case "attn_k_norm.weight" -> rmsK[layer] = floating(tensor.name, view);
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

        weights.qBias = concatenateFloat(qBias);
        weights.kBias = concatenateFloat(kBias);
        weights.vBias = concatenateFloat(vBias);

        weights.rmsAttWeight = concatenateFloat(rmsAtt);
        weights.rmsFfnWeight = concatenateFloat(rmsFfn);

        weights.postAttWeight = concatenateFloat(postAtt);
        weights.postFfnWeight = concatenateFloat(postFfn);

        weights.rmsQWeight = concatenateFloat(rmsQ);
        weights.rmsKWeight = concatenateFloat(rmsK);

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

    private static FloatTensor concatenateFloat(final FloatTensor[] parts) {
        if (Arrays.stream(parts).allMatch(java.util.Objects::isNull)) return null;
        return new CompositeFloatTensor(parts);
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
            case "token_embd.weight" -> weights.tokenEmbeddingTable = quantized(name, tensor);
            case "output.weight" -> weights.classifier = quantized(name, tensor);
            case "output_norm.weight" -> weights.rmsFinalWeight = floating(name, tensor);
            case "rope_factors_long.weight" -> weights.ropeFactorsLong = floating(name, tensor);
            case "rope_factors_short.weight" -> weights.ropeFactorsShort = floating(name, tensor);
            default -> { }
        }
    }

    private static QuantizedTensor quantized(final String name, final Tensor tensor) {
        if (tensor instanceof final QuantizedTensor q)
            return q;
        throw new IllegalArgumentException("Expected quantized tensor for %s, got %s"
                .formatted(name, tensor.getClass().getSimpleName()));
    }

    private static FloatTensor floating(final String name, final Tensor tensor) {
        if (tensor instanceof final FloatTensor f)
            return f;
        throw new IllegalArgumentException("Expected float tensor for %s, got %s"
                .formatted(name, tensor.getClass().getSimpleName()));
    }
}
