package com.epicmonstrosity.brewference.tokenizer.decoder;

public interface TokenDecoder {
    byte[] tokenToBytes(String token);
    String decode(String token);
}
