package com.epicmonstrosity.brewference.generation;

public interface TokenConsumer {
    void onPrefillTotal(int total);
    void onPrefillToken(int position, int tokenId, String tokenText);
    void onGeneratedToken(int position, int tokenId, String tokenText);
    void onComplete(GenerationResult result);
    void onDebug(String debug);
}
