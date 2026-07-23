package com.epicmonstrosity.brewference.model.smollm;

import com.epicmonstrosity.brewference.tokenizer.decoder.TokenDecoder;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class SmolLMTokenDecoder implements TokenDecoder {
    private final Map<Character, Byte> unicodeToByte;

    public SmolLMTokenDecoder(final char[] byteToUnicode) {
        this.unicodeToByte = new HashMap<>(256);
        for (int b = 0; b < 256; b++) {
            unicodeToByte.put(byteToUnicode[b], (byte) b);
        }
    }

    @Override
    public byte[] tokenToBytes(final String token) {
        final byte[] bytes = new byte[token.length()];
        for (int i = 0; i < token.length(); i++) {
            final Byte b = unicodeToByte.get(token.charAt(i));
            // Should never be null for a byte-level vocab; fail loudly if it is
            bytes[i] = (b != null) ? b : (byte) '?';
        }
        return bytes;
    }

    @Override
    public String decode(final String token) {
        return new String(tokenToBytes(token), StandardCharsets.UTF_8);
    }
}
