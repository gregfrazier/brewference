package com.epicmonstrosity.brewference.transformer.attention;

public class LayerContext {
    private final int tokenPosition;
    private final int layerNum;
    private final int layerOffset;
    private final long contextLength;

    public LayerContext(final int tokenPosition, final int layerNum, final int layerOffset, final long contextLength) {
        this.tokenPosition = tokenPosition;
        this.layerNum = layerNum;
        this.layerOffset = layerOffset;
        this.contextLength = contextLength;
    }

    public int getTokenPosition() {
        return tokenPosition;
    }

    public int getLayerNum() {
        return layerNum;
    }

    public int getLayerOffset() {
        return layerOffset;
    }

    public long getContextLength() {
        return contextLength;
    }
}
