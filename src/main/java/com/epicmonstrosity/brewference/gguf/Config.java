package com.epicmonstrosity.brewference.gguf;

import java.util.Map;

public class Config {
    private String filename;

    private int transformerDimensions;
    private int hiddenDimensions;
    private int numLayers;
    private int numHeads;
    private int numKVHeads;
    private int maxSequenceLength;

    private long slidingWindow;
    private Long headSize;
    private float layerNormRMSEpsilon;
    private long contextLength;
    private float ropeFrequencyBase;
    private float ropeFrequencyScale;

    private int vocabSize;
    private boolean addSepToken;
    private boolean addBosToken;
    private boolean addEosToken;
    private boolean addSpacePrefix;

    private String tokenizerModelType;
    private String architecture;
    private String sizeLabel;
    private String modelName;
    private String tokenizerPreTokenizer;

    // Token Ids
    private int bosToken;
    private int eosToken;
    private int unknownToken;
    private int sepToken;
    private int paddingToken;

    private Map<String, Object> metadata;

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Config setMetadata(final Map<String, Object> metadata) {
        this.metadata = metadata;
        return this;
    }

    public String getArchitecture() {
        return architecture;
    }

    public Config setArchitecture(final String architecture) {
        this.architecture = architecture;
        return this;
    }

    public String getTokenizerModelType() {
        return tokenizerModelType;
    }

    public Config setTokenizerModelType(final String tokenizerModelType) {
        this.tokenizerModelType = tokenizerModelType;
        return this;
    }


    public long getSlidingWindow() {
        return slidingWindow;
    }
    public Config setSlidingWindow(final long slidingWindow) {
        this.slidingWindow = slidingWindow;
        return this;
    }

    public Config setHeadSize(final long headSize) {
        this.headSize = headSize;
        return this;
    }

    public float getLayerNormRMSEpsilon() {
        return layerNormRMSEpsilon;
    }

    public Config setLayerNormRMSEpsilon(final float layerNormRMSEpsilon) {
        this.layerNormRMSEpsilon = layerNormRMSEpsilon;
        return this;
    }

    public long getContextLength() {
        return contextLength;
    }

    public Config setContextLength(final long contextLength) {
        this.contextLength = contextLength;
        return this;
    }

    public boolean isAddSepToken() {
        return addSepToken;
    }

    public Config setAddSepToken(final boolean addSepToken) {
        this.addSepToken = addSepToken;
        return this;
    }

    public float getRopeFrequencyBase() {
        return ropeFrequencyBase;
    }

    public Config setRopeFrequencyBase(final float ropeFrequencyBase) {
        this.ropeFrequencyBase = ropeFrequencyBase;
        return this;
    }

    public float getRopeFrequencyScale() {
        return ropeFrequencyScale;
    }

    public Config setRopeFrequencyScale(final float ropeFrequencyScale) {
        this.ropeFrequencyScale = ropeFrequencyScale;
        return this;
    }

    public boolean isAddBosToken() {
        return addBosToken;
    }

    public Config setAddBosToken(final boolean addBosToken) {
        this.addBosToken = addBosToken;
        return this;
    }

    public boolean isAddEosToken() {
        return addEosToken;
    }

    public Config setAddEosToken(final boolean addEosToken) {
        this.addEosToken = addEosToken;
        return this;
    }

    public int getHeadSize() {
        if (headSize != null)
            return headSize.intValue();
        return transformerDimensions / numHeads;
    }

    public boolean loadClassifier() {
        return vocabSize < 0;
    }

    public int getTransformerDimensions() {
        return transformerDimensions;
    }

    public Config setTransformerDimensions(final int transformerDimensions) {
        this.transformerDimensions = transformerDimensions;
        return this;
    }

    public int getHiddenDimensions() {
        return hiddenDimensions;
    }

    public Config setHiddenDimensions(final int hiddenDimensions) {
        this.hiddenDimensions = hiddenDimensions;
        return this;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public Config setNumLayers(final int numLayers) {
        this.numLayers = numLayers;
        return this;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public Config setNumHeads(final int numHeads) {
        this.numHeads = numHeads;
        return this;
    }

    public int getNumKVHeads() {
        return numKVHeads;
    }

    public Config setNumKVHeads(final int numKVHeads) {
        this.numKVHeads = numKVHeads;
        return this;
    }

    public int getVocabSize() {
        return Math.abs(vocabSize);
    }

    public Config setVocabSize(final int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public Config setMaxSequenceLength(final int maxSequenceLength) {
        this.maxSequenceLength = maxSequenceLength;
        return this;
    }

    public int getKeyValueDim() {
        return getNumKVHeads() * getHeadSize();
    }

    public int getKeyValueRatio() {
        return getNumHeads() / getNumKVHeads();
    }

    public int getQueryAttentionWidth() {
        return getNumHeads() * getHeadSize();
    }

    public String getSizeLabel() {
        return sizeLabel;
    }

    public Config setSizeLabel(final String sizeLabel) {
        this.sizeLabel = sizeLabel;
        return this;
    }

    public boolean isAddSpacePrefix() {
        return addSpacePrefix;
    }

    public Config setAddSpacePrefix(final boolean addSpacePrefix) {
        this.addSpacePrefix = addSpacePrefix;
        return this;
    }

    public String getModelName() {
        return modelName;
    }

    public Config setModelName(final String modelName) {
        this.modelName = modelName;
        return this;
    }

    @Override
    public String toString() {
        return "Config{" +
                "modelName='" + modelName + '\'' + "\n" +
                ", architecture='" + architecture + '\'' + "\n" +
                ", transformerDimensions=" + transformerDimensions + "\n" +
                ", hiddenDimensions=" + hiddenDimensions + "\n" +
                ", numLayers=" + numLayers + "\n" +
                ", numHeads=" + numHeads + "\n" +
                ", numKVHeads=" + numKVHeads + "\n" +
                ", vocabSize=" + vocabSize + "\n" +
                ", maxSequenceLength=" + maxSequenceLength + "\n" +
                ", slidingWindow=" + slidingWindow + "\n" +
                ", headSize=" + headSize + "\n" +
                ", layerNormRMSEpsilon=" + layerNormRMSEpsilon + "\n" +
                ", contextLength=" + contextLength + "\n" +
                ", addSepToken=" + addSepToken + "\n" +
                ", ropeFrequencyBase=" + ropeFrequencyBase + "\n" +
                ", ropeFrequencyScale=" + ropeFrequencyScale + "\n" +
                ", addBosToken=" + addBosToken + "\n" +
                ", addEosToken=" + addEosToken + "\n" +
                ", addSpacePrefix=" + addSpacePrefix + "\n" +
                ", sizeLabel='" + sizeLabel + '\'' + "\n" +
                ", tokenizerModelType='" + tokenizerModelType + '\'' + "\n" +
                ", preTokenizer='" + tokenizerPreTokenizer + '\'' + "\n" +
                '}';
    }

    public int getPaddingToken() {
        return paddingToken;
    }

    public Config setPaddingToken(final int paddingToken) {
        this.paddingToken = paddingToken;
        return this;
    }

    public int getSepToken() {
        return sepToken;
    }

    public Config setSepToken(final int sepToken) {
        this.sepToken = sepToken;
        return this;
    }

    public int getUnknownToken() {
        return unknownToken;
    }

    public Config setUnknownToken(final int unknownToken) {
        this.unknownToken = unknownToken;
        return this;
    }

    public int getEosToken() {
        return eosToken;
    }

    public Config setEosToken(final int eosToken) {
        this.eosToken = eosToken;
        return this;
    }

    public int getBosToken() {
        return bosToken;
    }

    public Config setBosToken(final int bosToken) {
        this.bosToken = bosToken;
        return this;
    }

    public String getTokenizerPreTokenizer() {
        return tokenizerPreTokenizer;
    }

    public Config setTokenizerPreTokenizer(final String tokenizerPreTokenizer) {
        this.tokenizerPreTokenizer = tokenizerPreTokenizer;
        return this;
    }

    public String getFilename() {
        return filename;
    }

    public Config setFilename(final String filename) {
        this.filename = filename;
        return this;
    }
}
