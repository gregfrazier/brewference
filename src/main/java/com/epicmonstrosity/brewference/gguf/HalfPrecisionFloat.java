package com.epicmonstrosity.brewference.gguf;

public final class HalfPrecisionFloat {
    private static final int UNSIGNED_SHORT_MASK = 0xFFFF;

    private static final int HALF_SIGN_SHIFT = 15;
    private static final int HALF_EXPONENT_SHIFT = 10;
    private static final int HALF_EXPONENT_MASK = 0x1F;
    private static final int HALF_MANTISSA_MASK = 0x3FF;
    private static final int HALF_NORMALIZATION_BIT = 0x400;
    private static final int HALF_EXPONENT_BIAS = 15;

    private static final int FLOAT_SIGN_SHIFT = 31;
    private static final int FLOAT_EXPONENT_SHIFT = 23;
    private static final int FLOAT_MANTISSA_SHIFT = 13;
    private static final int FLOAT_EXPONENT_BIAS = 127;
    private static final int FLOAT_INFINITY_EXPONENT = 0xFF;

    private HalfPrecisionFloat() { }

    static float toFloat(final short half) {
        final int halfBits = half & UNSIGNED_SHORT_MASK;
        final int signBits = ((halfBits >> HALF_SIGN_SHIFT) & 0x1) << FLOAT_SIGN_SHIFT;
        final int exponent = (halfBits >> HALF_EXPONENT_SHIFT) & HALF_EXPONENT_MASK;
        final int mantissa = halfBits & HALF_MANTISSA_MASK;

        final int floatBits;
        if (exponent == 0) {
            floatBits = mantissa == 0
                    ? signBits
                    : normalizedSubnormalFloatBits(signBits, mantissa);
        } else if (exponent == HALF_EXPONENT_MASK) {
            floatBits = signBits
                    | (FLOAT_INFINITY_EXPONENT << FLOAT_EXPONENT_SHIFT)
                    | (mantissa << FLOAT_MANTISSA_SHIFT);
        } else {
            floatBits = buildFloatBits(signBits, exponent, mantissa);
        }

        return Float.intBitsToFloat(floatBits);
    }

    private static int normalizedSubnormalFloatBits(final int signBits, int mantissa) {
        int exponent = 1;
        while ((mantissa & HALF_NORMALIZATION_BIT) == 0) {
            mantissa <<= 1;
            exponent--;
        }

        mantissa &= HALF_MANTISSA_MASK;
        return buildFloatBits(signBits, exponent, mantissa);
    }

    private static int buildFloatBits(final int signBits, final int halfExponent, final int mantissa) {
        final int floatExponent = halfExponent + FLOAT_EXPONENT_BIAS - HALF_EXPONENT_BIAS;
        return signBits
                | (floatExponent << FLOAT_EXPONENT_SHIFT)
                | (mantissa << FLOAT_MANTISSA_SHIFT);
    }
}
