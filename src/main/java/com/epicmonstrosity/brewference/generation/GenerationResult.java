package com.epicmonstrosity.brewference.generation;

public class GenerationResult {
    private final int promptTokenCount;
    private final int generatedTokenCount;
    private final boolean stoppedByEos;
    private final boolean stoppedByContextLimit;
    private final double elapsedNanos;

    public GenerationResult(
            final int promptTokenCount,
            final int generatedTokenCount,
            final boolean stoppedByEos,
            final boolean stoppedByContextLimit,
            final double elapsedNanos) {
        this.promptTokenCount = promptTokenCount;
        this.generatedTokenCount = generatedTokenCount;
        this.stoppedByEos = stoppedByEos;
        this.stoppedByContextLimit = stoppedByContextLimit;
        this.elapsedNanos = elapsedNanos;
    }

    public int getPromptTokenCount() {
        return promptTokenCount;
    }

    public int getGeneratedTokenCount() {
        return generatedTokenCount;
    }

    public boolean isStoppedByEos() {
        return stoppedByEos;
    }

    public boolean isStoppedByContextLimit() {
        return stoppedByContextLimit;
    }

    public double getElapsedNanos() {
        return elapsedNanos;
    }
}
