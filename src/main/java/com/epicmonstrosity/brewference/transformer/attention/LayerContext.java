package com.epicmonstrosity.brewference.transformer.attention;

public class LayerContext {
    private final int tokenPosition;
    private final int layerNum;
    private final int layerOffset;

    public LayerContext(final int tokenPosition, final int layerNum, final int layerOffset) {
        this.tokenPosition = tokenPosition;
        this.layerNum = layerNum;
        this.layerOffset = layerOffset;
    }

    public int getTokenPosition() {
        return tokenPosition;
    }

    public int getLayerNum() {
        return layerNum;
    }

    /**
     * KV cache layer offset
     * <p>
     * [layer][position][kv_dim]
     */
    public int getLayerOffset() {
        return layerOffset;
    }
}
