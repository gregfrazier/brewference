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
    
    /**
     * Computes the dot product between a quantized data block and a floating-point array.
     * The method processes the quantized data in blocks of 32 elements, applying block-specific
     * scaling factors to compute the final result.
     *
     * @param q The Q8Block containing the quantized data and corresponding scaling factors.
     *          Each block in the Q8Block corresponds to 32 elements.
     * @param qOffset The starting offset within the quantized data array of the Q8Block.
     *                This determines the initial position for reading quantized values.
     * @param x The floating-point array with which to compute the dot product.
     *          It must have enough elements starting from the specified offset.
     * @param xOffset The starting offset within the floating-point array.
     *                This determines the initial position for reading floating-point values.
     * @param n The number of elements to process for the dot product.
     *          This value must be a multiple of 32 to align with the block structure.
     * @return The computed dot product value as a floating-point number.
     */
    public float dot(final Q8Block q, final int qOffset, final float[] x, final int xOffset, final int n) {
        float sum = 0.0f;
        final int numBlocks = n / 32;
        final int blockIdx = qOffset / 32;
        for (int b = 0; b < numBlocks; b++) {
            final float scale = q.scales[blockIdx + b];
            float blockSum = 0.0f;
            for (int i = 0; i < 32; i++) {
                blockSum += q.data[qOffset + b * 32 + i] * x[xOffset + b * 32 + i];
            }
            sum += blockSum * scale;
        }
        return sum;
    }
}
