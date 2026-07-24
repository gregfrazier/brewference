package com.epicmonstrosity.brewference.tokenizer.decoder;

public class SimpleTokenDecoder implements TokenDecoder {
    @Override
    public byte[] tokenToBytes(final String token) {
        return decode(token).getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }

    @Override
    public String decode(final String token) {
        //if (vocab.getSpecialTokens().contains(token)) {
        //    return "\n";
        //}

        return token == null
                ? "[UNKNOWN]"
                : token
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
                .replace("<0x0D>", "\r")
                .replace("<0x09>", "\t");
    }
}
