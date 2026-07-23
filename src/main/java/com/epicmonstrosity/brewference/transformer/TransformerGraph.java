package com.epicmonstrosity.brewference.transformer;

import com.epicmonstrosity.brewference.gguf.QuantizedWeights;

public interface TransformerGraph {
    /**
     * Executes a forward pass through the transformer model, computing the intermediate
     * and output states for the given input token and position.
     *
     * @param token The input token represented as an integer. Usually corresponds to a token ID in the vocabulary.
     * @param pos The position of the token in the sequence, indexed starting from 0.
     * @param runState The mutable state object that holds intermediate activations and buffer states
     *                 used throughout the forward pass. This state is updated during the computation.
     * @param weights The quantized weight parameters of the transformer model, including embeddings,
     *                attention weights, feedforward weights, and related parameters.
     */
    void forward(int token, int pos, RunState runState, QuantizedWeights weights);
}
