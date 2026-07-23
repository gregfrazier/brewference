package com.epicmonstrosity.brewference.model.qwen2;

import com.epicmonstrosity.brewference.tokenizer.decoder.TokenDecoder;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class Qwen2TokenDecoder implements TokenDecoder {
    private final Map<Character, Byte> unicodeToByte;

    public Qwen2TokenDecoder(final char[] byteToUnicode) {
        this.unicodeToByte = new HashMap<>(256);
        for (int byteIndex = 0; byteIndex < 256; byteIndex++) {
            unicodeToByte.put(byteToUnicode[byteIndex], (byte) byteIndex);
        }
    }

    public byte[] tokenToBytes(final String token) {
        final byte[] bytes = new byte[token.length()];
        for (int i = 0; i < token.length(); i++) {
            final Byte b = unicodeToByte.get(token.charAt(i));
            // Should never be null for a byte-level vocab
            bytes[i] = (b != null) ? b : (byte) '?';
        }
        return bytes;
    }

    public String decode(final String token) {
        return new String(tokenToBytes(token), StandardCharsets.UTF_8);
    }
}
