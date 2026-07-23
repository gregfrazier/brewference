package com.epicmonstrosity.brewference.gguf;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class GgufWeightsLoader {
    private final Reader ggufReader;
    private final FileChannel channel;
    protected final QuantizedWeights weights = new QuantizedWeights();

    public GgufWeightsLoader(final Reader ggufReader, final FileChannel channel) {
        this.ggufReader = ggufReader;
        this.channel = channel;
    }

    // This code sucks and needs to be refactored
    public QuantizedWeights load(final Config config) throws IOException {
        final int numLayers = config.getNumLayers();

        final QuantizedWeights.Q8Block[] wqByLayer = new QuantizedWeights.Q8Block[numLayers];
        final QuantizedWeights.Q8Block[] wkByLayer = new QuantizedWeights.Q8Block[numLayers];
        final QuantizedWeights.Q8Block[] wvByLayer = new QuantizedWeights.Q8Block[numLayers];

        final float[][] wqBiasByLayer = new float[numLayers][];
        final float[][] wkBiasByLayer = new float[numLayers][];
        final float[][] wvBiasByLayer = new float[numLayers][];

        final QuantizedWeights.Q8Block[] woByLayer = new QuantizedWeights.Q8Block[numLayers];
        final QuantizedWeights.Q8Block[] w1ByLayer = new QuantizedWeights.Q8Block[numLayers];
        final QuantizedWeights.Q8Block[] w2ByLayer = new QuantizedWeights.Q8Block[numLayers];
        final QuantizedWeights.Q8Block[] w3ByLayer = new QuantizedWeights.Q8Block[numLayers];

        final float[][] rmsAttByLayer = new float[numLayers][];
        final float[][] rmsFfnByLayer = new float[numLayers][];
        final float[][] postAttByLayer = new float[numLayers][];
        final float[][] postFfnByLayer = new float[numLayers][];
        final float[][] rmsQByLayer = new float[numLayers][];
        final float[][] rmsKByLayer = new float[numLayers][];

        final Pattern blkPat = Pattern.compile("^blk\\.(\\d+)\\.(.+)$");

        for (final Reader.TensorInfo tensor : ggufReader.getTensors()) {
            final long offset = ggufReader.getDataOffset() + tensor.offset;

            int size = 1;
            for (final long d : tensor.dims) {
                if (d <= 0 || d > Integer.MAX_VALUE) {
                    throw new IOException("Bad tensor dim in " + tensor.name + ": " + d);
                }
                size *= (int) d;
            }

            channel.position(offset);

            switch (tensor.name) {
                case "token_embd.weight":
                    weights.tokenEmbeddingTable = readQ8Block(channel, size, tensor.name);
                    continue;
                case "output_norm.weight":
                    weights.rmsFinalWeight = readFloatArray(channel, size, tensor.name);
                    continue;
                case "output.weight":
                    weights.classifier = readQ8Block(channel, size, tensor.name);
                    continue;
            }

            final Matcher m = blkPat.matcher(tensor.name);
            if (!m.matches()) {
                continue;
            }

            final int layer = Integer.parseInt(m.group(1));
            final String tail = m.group(2);

            if (layer < 0 || layer >= numLayers) {
                throw new IOException("Tensor has out-of-range layer index: " + tensor.name);
            }

            //System.out.println("Processing tensor: " + tensor.name);
            switch (tail) {
                case "attn_q.weight":
                    wqByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "attn_k.weight":
                    wkByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "attn_v.weight":
                    wvByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "attn_q.bias":
                    wqBiasByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "attn_k.bias":
                    wkBiasByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "attn_v.bias":
                    wvBiasByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "attn_output.weight":
                    woByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "ffn_gate.weight":
                    w1ByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "ffn_down.weight":
                    w2ByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "ffn_up.weight":
                    w3ByLayer[layer] = readQ8Block(channel, size, tensor.name);
                    break;
                case "attn_norm.weight":
                    rmsAttByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "ffn_norm.weight":
                    rmsFfnByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "post_attention_norm.weight":
                    postAttByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "post_ffw_norm.weight":
                    postFfnByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "attn_q_norm.weight":
                    rmsQByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
                case "attn_k_norm.weight":
                    rmsKByLayer[layer] = readFloatArray(channel, size, tensor.name);
                    break;
            }
        }

        weights.wq = concatenateQ8Blocks(java.util.Arrays.asList(wqByLayer));
        weights.wk = concatenateQ8Blocks(java.util.Arrays.asList(wkByLayer));
        weights.wv = concatenateQ8Blocks(java.util.Arrays.asList(wvByLayer));
        weights.wo = concatenateQ8Blocks(java.util.Arrays.asList(woByLayer));
        weights.w1 = concatenateQ8Blocks(java.util.Arrays.asList(w1ByLayer));
        weights.w2 = concatenateQ8Blocks(java.util.Arrays.asList(w2ByLayer));
        weights.w3 = concatenateQ8Blocks(java.util.Arrays.asList(w3ByLayer));

        // For Qwen2
        weights.qBias = concatenateFloatArrays(java.util.Arrays.asList(wqBiasByLayer));
        weights.kBias = concatenateFloatArrays(java.util.Arrays.asList(wkBiasByLayer));
        weights.vBias = concatenateFloatArrays(java.util.Arrays.asList(wvBiasByLayer));

        weights.rmsAttWeight = concatenateFloatArrays(java.util.Arrays.asList(rmsAttByLayer));
        weights.rmsFfnWeight = concatenateFloatArrays(java.util.Arrays.asList(rmsFfnByLayer));
        weights.postAttWeight = concatenateFloatArrays(java.util.Arrays.asList(postAttByLayer));
        weights.postFfnWeight = concatenateFloatArrays(java.util.Arrays.asList(postFfnByLayer));
        weights.rmsQWeight = concatenateFloatArrays(java.util.Arrays.asList(rmsQByLayer));
        weights.rmsKWeight = concatenateFloatArrays(java.util.Arrays.asList(rmsKByLayer));

        if (weights.classifier == null) {
            weights.classifier = weights.tokenEmbeddingTable;
            // System.out.println("Using tied weights: classifier = tokenEmbeddingTable");
        }

        return weights;
    }

    protected QuantizedWeights.Q8Block concatenateQ8Blocks(final java.util.List<QuantizedWeights.Q8Block> blocks) {
        if (blocks.isEmpty()) {
            return null;
        }

        final int totalSize = calculateTotalSize(blocks);
        final QuantizedWeights.Q8Block result = new QuantizedWeights.Q8Block(totalSize);

        int dataOffset = 0;
        int scaleOffset = 0;
        for (final QuantizedWeights.Q8Block block : blocks) {
            System.arraycopy(block.data, 0, result.data, dataOffset, block.data.length);
            System.arraycopy(block.scales, 0, result.scales, scaleOffset, block.scales.length);
            dataOffset += block.data.length;
            scaleOffset += block.scales.length;
        }

        return result;
    }

    private static int calculateTotalSize(final List<QuantizedWeights.Q8Block> blocks) {
        int totalSize = 0;
        for (final QuantizedWeights.Q8Block block : blocks) {
            if (block == null) {
                throw new IllegalArgumentException("Null Q8Block in list");
            }
            if (block.data.length % 32 != 0) {
                throw new IllegalArgumentException("Q8Block data length must be multiple of 32, got " + block.data.length);
            }
            if (block.scales.length != block.data.length / 32) {
                throw new IllegalArgumentException("Q8Block scales length mismatch: scales=" +
                        block.scales.length + " data/32=" + (block.data.length / 32));
            }
            totalSize += block.data.length;
        }
        return totalSize;
    }

    protected float[] concatenateFloatArrays(final java.util.List<float[]> arrays) {
        if (arrays.isEmpty() || arrays.stream().allMatch(Objects::isNull)) {
            return null;
        }

        int totalSize = 0;
        for (final float[] arr : arrays) {
            if (arr == null) {
                throw new IllegalArgumentException("Null float[] in list");
            }
            totalSize += arr.length;
        }

        final float[] result = new float[totalSize];
        int offset = 0;
        for (final float[] arr : arrays) {
            System.arraycopy(arr, 0, result, offset, arr.length);
            offset += arr.length;
        }

        return result;
    }

    protected QuantizedWeights.Q8Block readQ8Block(final FileChannel channel, final int size, final String tensorName) throws IOException {
        if (size % 32 != 0) {
            throw new IOException("Tensor " + tensorName + " has " + size + " elements, not divisible by 32");
        }

        final QuantizedWeights.Q8Block block = new QuantizedWeights.Q8Block(size);
        final int numBlocks = size / 32;
        final int bytesToRead = numBlocks * 34;

        final ByteBuffer buf = ByteBuffer.allocate(bytesToRead).order(ByteOrder.LITTLE_ENDIAN);
        readTensorData(channel, buf, tensorName);
        buf.flip();

        for (int i = 0; i < numBlocks; i++) {
            block.scales[i] = HalfPrecisionFloat.toFloat(buf.getShort());
            buf.get(block.data, i * 32, 32);
        }

        return block;
    }

    protected float[] readFloatArray(final FileChannel channel, final int size, final String name) throws IOException {
        final Reader.TensorInfo info = ggufReader.getTensor(name);
        if (info == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }

        final float[] result = new float[size];

        if (info.type == 0) {
            final ByteBuffer buf = ByteBuffer.allocate(size * 4).order(ByteOrder.LITTLE_ENDIAN);
            readTensorData(channel, buf, name);
            buf.flip();

            for (int i = 0; i < size; i++) {
                result[i] = buf.getFloat();
            }
        } else if (info.type == 1) {
            final ByteBuffer buf = ByteBuffer.allocate(size * 2).order(ByteOrder.LITTLE_ENDIAN);
            readTensorData(channel, buf, name);
            buf.flip();

            for (int i = 0; i < size; i++) {
                result[i] = HalfPrecisionFloat.toFloat(buf.getShort());
            }
        } else {
            throw new IOException("Unexpected tensor type " + info.type + " for " + name);
        }

        return result;
    }

    protected void readTensorData(final FileChannel channel, final ByteBuffer buf, final String tensorName) throws IOException {
        while (buf.hasRemaining()) {
            if (channel.read(buf) == -1) {
                throw new EOFException("Reached end of stream while reading " + tensorName);
            }
        }
    }
}
