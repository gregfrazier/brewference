package com.epicmonstrosity.brewference.generation;

public class GenerationOptions {
    private int maxNewTokens = 256;
    private float temperature = 1.0f;
    private float topP = 0.95f;
    private int topK = 40;
    private float repetitionPenalty = 1.0f;
    private boolean echoPrompt = false;

    public GenerationOptions() { }

    public int getMaxNewTokens() {
        return maxNewTokens;
    }

    public GenerationOptions setMaxNewTokens(int maxNewTokens) {
        this.maxNewTokens = maxNewTokens;
        return this;
    }

    public float getTemperature() {
        return temperature;
    }

    public GenerationOptions setTemperature(float temperature) {
        this.temperature = temperature;
        return this;
    }

    public float getTopP() {
        return topP;
    }

    public GenerationOptions setTopP(float topP) {
        this.topP = topP;
        return this;
    }

    public int getTopK() {
        return topK;
    }

    public GenerationOptions setTopK(int topK) {
        this.topK = topK;
        return this;
    }

    public float getRepetitionPenalty() {
        return repetitionPenalty;
    }

    public GenerationOptions setRepetitionPenalty(float repetitionPenalty) {
        this.repetitionPenalty = repetitionPenalty;
        return this;
    }

    public boolean isEchoPrompt() {
        return echoPrompt;
    }

    public GenerationOptions setEchoPrompt(boolean echoPrompt) {
        this.echoPrompt = echoPrompt;
        return this;
    }
}
