package com.epicmonstrosity.brewference.transformer.ffn;

import com.epicmonstrosity.brewference.gguf.Config;
import com.epicmonstrosity.brewference.transformer.RunState;

import java.util.function.DoubleUnaryOperator;

public final class FfnActivation {
    private static final float SQRT_TWO_OVER_PI = 0.7978845608028654f;
    private static final float GELU_COEFFICIENT = 0.044715f;

    private FfnActivation() {}

    /**
     * Applies the Gated Linear Unit (Gelu) activation function using PyTorch-style approximation
     * (tanh-based GELU) to the hidden state of the transformer forward pass.
     *
     * @param runState The mutable state for the transformer forward pass, containing the
     *                 activation buffers, including the hidden state arrays to be modified.
     * @param config   The configuration object containing details such as the dimensions
     *                 of the model, required to process the activation.
     */
    public static void geGlu(final RunState runState, final Config config) {
        applyGatedActivation(runState, config, FfnActivation::geluPytorchTanh);
    }

    /**
     * Applies the SwiGLU (Sigmoid-weighted Linear Unit) activation function
     * to the hidden state of the transformer forward pass. This operation
     * combines the Sigmoid (SiLU) activation with element-wise multiplication
     * using secondary hidden state buffers to produce the desired transformation.
     *
     * @param runState The mutable state for the transformer forward pass, containing
     *                 the activation buffers, including the hidden state arrays
     *                 to be modified.
     * @param config   The configuration object containing details such as the dimensions
     *                 of the model, required to process the activation.
     */
    public static void swiGlu(final RunState runState, final Config config) {
        applyGatedActivation(runState, config, FfnActivation::silu);
    }

    private static void applyGatedActivation(final RunState runState, final Config config, final DoubleUnaryOperator activation) {
        final int hiddenDimensions = config.getHiddenDimensions();
        for (int i = 0; i < hiddenDimensions; i++) {
            final float value = runState.hb[i];
            runState.hb[i] = (float) activation.applyAsDouble(value) * runState.hb2[i];
        }
    }

    private static double geluPytorchTanh(final double value) {
        final double inner = SQRT_TWO_OVER_PI * (value + GELU_COEFFICIENT * value * value * value);
        return 0.5 * value * (1.0 + Math.tanh(inner));
    }

    private static double silu(final double value) {
        return value / (1.0 + Math.exp(-value));
    }
}
