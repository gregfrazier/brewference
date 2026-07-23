package com.epicmonstrosity.brewference.gguf;

public class QuantizedWeights {
    /**
     * Represents a block of quantized data used in quantization-related computations.
     * <p>
     * The Q8Block class is designed to store blocks of quantized data along with their
     * corresponding scaling factors. The quantized data is stored in a byte array to minimize
     * memory usage, while the scaling factors are stored in a separate float array for each block.
     * Each block corresponds to 32 elements.
     * <p>
     * Fields:
     * - scales: A float array containing the scaling factors for each block of quantized data.
     *           The size of this array depends on the number of blocks, which is derived from
     *           the total number of quantized elements divided by 32.
     * - data: A byte array containing the actual quantized data. Its size is equal to the
     *         total number of quantized elements.
     * <p>
     * Constructor:
     * - Q8Block(int numElements): Initializes an instance of the Q8Block class.
     *   - numElements: The total number of quantized elements to be stored in the block.
     *                  Must be a multiple of 32 to align with the block structure.
     *   - This constructor calculates the required number of blocks based on the input size
     *     and initializes the scale and data arrays accordingly.
     */
    public static class Q8Block {
        public float[] scales;
        public final byte[] data;

        public Q8Block(final int numElements) {
            final int numBlocks = numElements / 32;
            this.scales = new float[numBlocks];
            this.data = new byte[numElements];
        }
    }

    public Q8Block tokenEmbeddingTable;
    public Q8Block wq, wk, wv, wo;
    public Q8Block w1, w2, w3;
    public Q8Block classifier;

    // Bias is float
    public float[] qBias;
    public float[] kBias;
    public float[] vBias;

    // Norms are always float
    public float[] rmsAttWeight;
    public float[] rmsFfnWeight;
    public float[] rmsFinalWeight;

    public float[] postAttWeight;
    public float[] postFfnWeight;

    public float[] rmsKWeight;
    public float[] rmsQWeight;
    
    // RoPE is float
    public float[] freqCisReal;
    public float[] freqCisImag;

    /**
     * Dequantizes a row of quantized data from a Q8Block and writes the resulting floating-point
     * values into an output array.
     *
     * @param q The Q8Block containing the quantized data and the corresponding scale factors.
     *          It is assumed that each block corresponds to 32 elements and that the scale
     *          factors represent the magnitude of quantization for each block.
     * @param rowOffset The starting offset within the Q8Block's data array, indicating the first
     *                  element of the row to dequantize.
     * @param n The total number of elements to dequantize. This value must be a multiple of 32,
     *          as it aligns with the block structure of the Q8Block.
     * @param out The output array where the dequantized floating-point values are written. It is
     *            assumed to have enough capacity to store all the dequantized elements.
     */
    public void dequantizeRow(final Q8Block q, final int rowOffset, final int n, final float[] out) {
        final int numBlocks = n / 32;
        final int blockOffset = rowOffset / 32;
        for (int b = 0; b < numBlocks; b++) {
            final float scale = q.scales[blockOffset + b];
            for (int i = 0; i < 32; i++) {
                out[b * 32 + i] = q.data[rowOffset + b * 32 + i] * scale;
            }
        }
    }
}
