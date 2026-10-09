package com.epicmonstrosity.brewference.gguf.data;

import com.epicmonstrosity.brewference.template.chat.ChatRole;

import java.util.List;
import java.util.Map;

public class Config {
    private String filename;

    private int transformerDimensions;
    private int hiddenDimensions;
    private int numLayers;
    private int numHeads;
    private int numKVHeads;
    private int maxSequenceLength;
    private int embeddingLengthPerLayerInput;

    private long slidingWindow;
    private Long headSize;
    private float layerNormRMSEpsilon;
    private long contextLength;
    private float ropeFrequencyBase;
    private float ropeFrequencyScale;
    private float ropeScalingFactor;

    private int ropeScalingOriginalContextLength;
    private float ropeScalingAttnFactor;
    private int ropeDimCount;

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

    private int attentionKeyLength;

    // SSM / Gated Delta Net
    private int fullAttentionInterval;
    private boolean[] recurrentLayers;
    private int convKernel;
    private int stateSize;
    private int groupCount;
    private int timeStepRank;
    private int innerSize;

    // Convenience
    private List<String> tensorNames;

    private Map<String, Object> metadata;

    private Map<ChatRole, String> roleMapping;

    // Jinja Template
    private String jinjaTemplate;

    /**
     * Raw chat template from {@code tokenizer.chat_template}, in Jinja syntax.
     * Null when the checkpoint ships no template, in which case the caller falls back to a
     * built-in prompt template.
     */
    public String getJinjaTemplate() {
        return jinjaTemplate;
    }

    public Config setJinjaTemplate(final String jinjaTemplate) {
        this.jinjaTemplate = jinjaTemplate;
        return this;
    }

    /**
     * The full GGUF metadata map as read from the checkpoint, keyed by GGUF metadata key.
     * <p>
     * Holds every key, including the ones with no {@code Config} field (MoE parameters, merges,
     * softcapping), so per-model loaders and the tokenizer loaders can read them directly.
     * The live map is returned, not a copy.
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Config setMetadata(final Map<String, Object> metadata) {
        this.metadata = metadata;
        return this;
    }

    /**
     * {@code general.architecture} - the architecture id, e.g. {@code qwen35}, {@code llama},
     * {@code gemma3}.
     * <p>
     * Selects the model implementation in the runner factory and prefixes every
     * {@code <arch>.*} metadata lookup during parsing.
     */
    public String getArchitecture() {
        return architecture;
    }

    public Config setArchitecture(final String architecture) {
        this.architecture = architecture;
        return this;
    }

    /**
     * {@code tokenizer.ggml.model} - the tokenizer algorithm family, e.g. {@code gpt2},
     * {@code llama-bpe}, {@code spm}.
     * <p>
     * One half of {@link #getCodecId()}; the other is {@link #getTokenizerPreTokenizer()}.
     */
    public String getTokenizerModelType() {
        return tokenizerModelType;
    }

    public Config setTokenizerModelType(final String tokenizerModelType) {
        this.tokenizerModelType = tokenizerModelType;
        return this;
    }


    /**
     * {@code <arch>.attention.sliding_window} - how many previous tokens a local (sliding-window)
     * attention layer may look back at.
     * <p>
     * 0 when the key is absent, which attention patterns read as "no windowing, attend globally".
     */
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

    /**
     * {@code <arch>.attention.layer_norm_rms_epsilon} - the epsilon added inside the RMSNorm
     * denominator to keep it away from zero.
     */
    public float getLayerNormRMSEpsilon() {
        return layerNormRMSEpsilon;
    }

    public Config setLayerNormRMSEpsilon(final float layerNormRMSEpsilon) {
        this.layerNormRMSEpsilon = layerNormRMSEpsilon;
        return this;
    }

    /**
     * {@code <arch>.context_length} exactly as the checkpoint states it.
     * <p>
     * Informational only: the architecture cannot always honour the advertised length, so
     * allocation and generation limits use {@link #getMaxSequenceLength()} instead.
     */
    public long getContextLength() {
        return contextLength;
    }

    public Config setContextLength(final long contextLength) {
        this.contextLength = contextLength;
        return this;
    }

    /**
     * {@code tokenizer.ggml.add_sep_token} - whether the tokenizer inserts a separator token
     * ({@link #getSepToken()}) between segments.
     */
    public boolean isAddSepToken() {
        return addSepToken;
    }

    public Config setAddSepToken(final boolean addSepToken) {
        this.addSepToken = addSepToken;
        return this;
    }

    /**
     * {@code <arch>.rope.freq_base} - the base of the geometric frequency series used to build
     * rotary angles. Absent means the implementation's own default applies.
     */
    public float getRopeFrequencyBase() {
        return ropeFrequencyBase;
    }

    public Config setRopeFrequencyBase(final float ropeFrequencyBase) {
        this.ropeFrequencyBase = ropeFrequencyBase;
        return this;
    }

    /**
     * Multiplier applied to the token position before rotary angles are computed
     * ({@code <arch>.rope.frequency_scale}).
     * <p>
     * Rope implementations treat any value {@code <= 0} as unset and fall back to 1.0.
     */
    public float getRopeFrequencyScale() {
        return ropeFrequencyScale;
    }

    public Config setRopeFrequencyScale(final float ropeFrequencyScale) {
        this.ropeFrequencyScale = ropeFrequencyScale;
        return this;
    }

    /**
     * {@code <arch>.rope.scaling.factor} - the YaRN-style position interpolation factor.
     * <p>
     * Global-attention layers in Gemma 3 use its reciprocal as an attention logit scale; a value
     * {@code <= 0} means no rope scaling is configured.
     */
    public float getRopeScalingFactor() {
        return ropeScalingFactor;
    }

    public Config setRopeScalingFactor(final float ropeScalingFactor) {
        this.ropeScalingFactor = ropeScalingFactor;
        return this;
    }

    /**
     * {@code tokenizer.ggml.add_bos_token} - whether a beginning-of-sequence token
     * ({@link #getBosToken()}) is prepended to every prompt.
     */
    public boolean isAddBosToken() {
        return addBosToken;
    }

    public Config setAddBosToken(final boolean addBosToken) {
        this.addBosToken = addBosToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.add_eos_token} - whether an end-of-sequence token
     * ({@link #getEosToken()}) is appended to every prompt.
     */
    public boolean isAddEosToken() {
        return addEosToken;
    }

    public Config setAddEosToken(final boolean addEosToken) {
        this.addEosToken = addEosToken;
        return this;
    }

    /**
     * Width of a single attention head, in elements.
     * <p>
     * Per-model checkpoint loaders set this explicitly when the architecture deviates from the
     * naive split (Gemma 2/3: 256, Qwen3: 128). When no loader assigned it, it falls back to
     * {@code transformerDimensions / numHeads}.
     */
    public int getHeadSize() {
        if (headSize != null)
            return headSize.intValue();
        return transformerDimensions / numHeads;
    }

    /**
     * Classifier-mode flag: true when the checkpoint recorded a negative vocabulary size.
     * <p>
     * The sign is metadata only - {@link #getVocabSize()} strips it either way - and no loader
     * currently branches on this flag.
     */
    public boolean loadClassifier() {
        return vocabSize < 0;
    }

    /**
     * {@code <arch>.embedding_length} - the residual stream width (model dimension) that every
     * layer input and output is shaped to.
     */
    public int getTransformerDimensions() {
        return transformerDimensions;
    }

    public Config setTransformerDimensions(final int transformerDimensions) {
        this.transformerDimensions = transformerDimensions;
        return this;
    }

    /**
     * {@code <arch>.feed_forward_length} - the intermediate width of the feed-forward block,
     * i.e. the width of the w1/w3 outputs and the w2 input.
     */
    public int getHiddenDimensions() {
        return hiddenDimensions;
    }

    public Config setHiddenDimensions(final int hiddenDimensions) {
        this.hiddenDimensions = hiddenDimensions;
        return this;
    }

    /**
     * {@code <arch>.block_count} - number of transformer layers. Layer indices run
     * {@code 0 .. numLayers - 1}, and this is the expected length of {@link #getRecurrentLayers()}.
     */
    public int getNumLayers() {
        return numLayers;
    }

    public Config setNumLayers(final int numLayers) {
        this.numLayers = numLayers;
        return this;
    }

    /**
     * {@code <arch>.attention.head_count} - number of query heads.
     */
    public int getNumHeads() {
        return numHeads;
    }

    public Config setNumHeads(final int numHeads) {
        this.numHeads = numHeads;
        return this;
    }

    /**
     * {@code <arch>.attention.head_count_kv} - number of key/value heads.
     * <p>
     * Equal to {@link #getNumHeads()} for multi-head attention, smaller for grouped-query
     * attention (see {@link #getKeyValueRatio()}).
     */
    public int getNumKVHeads() {
        return numKVHeads;
    }

    public Config setNumKVHeads(final int numKVHeads) {
        this.numKVHeads = numKVHeads;
        return this;
    }

    /**
     * Number of vocabulary entries, taken from the length of {@code tokenizer.ggml.tokens}.
     * <p>
     * Always positive: the stored value keeps its sign so a negative count can flag classifier
     * mode (see {@link #loadClassifier()}). This is the output width of the classifier matmul.
     */
    public int getVocabSize() {
        return Math.abs(vocabSize);
    }

    public Config setVocabSize(final int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    /**
     * Sequence length the runtime actually honours, and the one every KV cache and run-state
     * buffer is allocated for.
     * <p>
     * Seeded from {@code <arch>.context_length} during parsing but writable by the user, so it -
     * never {@link #getContextLength()} - is what generation limits and buffer offsets use.
     */
    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public Config setMaxSequenceLength(final int maxSequenceLength) {
        this.maxSequenceLength = maxSequenceLength;
        return this;
    }

    /**
     * Total width of the key and value projections: {@code numKVHeads * headSize}.
     * <p>
     * Used both as the matmul output width for w_k/w_v and as the per-token stride of the KV cache.
     */
    public int getKeyValueDim() {
        return getNumKVHeads() * getHeadSize();
    }

    /**
     * How many query heads share one key/value head: {@code numHeads / numKVHeads}.
     * <p>
     * 1 means multi-head attention; anything greater means grouped-query attention.
     */
    public int getKeyValueRatio() {
        return getNumHeads() / getNumKVHeads();
    }

    /**
     * Total width of the query projection, and of the output projection that folds the heads back
     * into the residual stream: {@code numHeads * headSize}.
     */
    public int getQueryAttentionWidth() {
        return getNumHeads() * getHeadSize();
    }

    /**
     * {@code general.size_label} - the human-readable parameter count, e.g. {@code 7B}.
     * Display only; nothing in the runtime depends on it.
     */
    public String getSizeLabel() {
        return sizeLabel;
    }

    public Config setSizeLabel(final String sizeLabel) {
        this.sizeLabel = sizeLabel;
        return this;
    }

    /**
     * {@code tokenizer.ggml.add_space_prefix} - whether the tokenizer prefixes pieces with a space,
     * which changes how a prompt must be split before encoding.
     */
    public boolean isAddSpacePrefix() {
        return addSpacePrefix;
    }

    public Config setAddSpacePrefix(final boolean addSpacePrefix) {
        this.addSpacePrefix = addSpacePrefix;
        return this;
    }

    /**
     * {@code general.name} - the model's display name as recorded in the checkpoint.
     * Display only; architecture dispatch uses {@link #getArchitecture()}.
     */
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
                ", headSize=" + getHeadSize() + "\n" +
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

    /**
     * {@code tokenizer.ggml.padding_token_id} - id of the padding token.
     * <p>
     * 0 when the key is absent, which is indistinguishable from a checkpoint that genuinely uses
     * id 0 as its pad token.
     */
    public int getPaddingToken() {
        return paddingToken;
    }

    public Config setPaddingToken(final int paddingToken) {
        this.paddingToken = paddingToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.separator_token_id} - id of the separator token inserted between
     * segments when {@link #isAddSepToken()} is set.
     * <p>
     * 0 when the key is absent.
     */
    public int getSepToken() {
        return sepToken;
    }

    public Config setSepToken(final int sepToken) {
        this.sepToken = sepToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.unknown_token_id} - id substituted for a piece the vocabulary cannot
     * represent.
     * <p>
     * 0 when the key is absent.
     */
    public int getUnknownToken() {
        return unknownToken;
    }

    public Config setUnknownToken(final int unknownToken) {
        this.unknownToken = unknownToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.eos_token_id} - id that ends generation; the runner stops as soon as it
     * is sampled.
     * <p>
     * 0 when the key is absent, which would read as "token 0 stops generation".
     */
    public int getEosToken() {
        return eosToken;
    }

    public Config setEosToken(final int eosToken) {
        this.eosToken = eosToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.bos_token_id} - id prepended to the prompt when
     * {@link #isAddBosToken()} is set.
     * <p>
     * 0 when the key is absent.
     */
    public int getBosToken() {
        return bosToken;
    }

    public Config setBosToken(final int bosToken) {
        this.bosToken = bosToken;
        return this;
    }

    /**
     * {@code tokenizer.ggml.pre} - the pre-tokenizer variant, e.g. {@code qwen2} or {@code llama3},
     * which decides how text is split into pieces before vocabulary lookup.
     * <p>
     * One half of {@link #getCodecId()}; the other is {@link #getTokenizerModelType()}.
     */
    public String getTokenizerPreTokenizer() {
        return tokenizerPreTokenizer;
    }

    public Config setTokenizerPreTokenizer(final String tokenizerPreTokenizer) {
        this.tokenizerPreTokenizer = tokenizerPreTokenizer;
        return this;
    }

    /**
     * Path of the GGUF file this config was read from, set by the checkpoint loader.
     */
    public String getFilename() {
        return filename;
    }

    public Config setFilename(final String filename) {
        this.filename = filename;
        return this;
    }

    /**
     * {@code <preTokenizer>-<tokenizerModelType>} - the key the prompt-encoder registry uses to find
     * the token codec that matches this checkpoint's tokenizer.
     * <p>
     * Not a GGUF value: it is composed from {@link #getTokenizerPreTokenizer()} and
     * {@link #getTokenizerModelType()}, so either being unset yields an id that will not resolve to
     * a codec.
     */
    public String getCodecId() {
        return String.format("%s-%s", tokenizerPreTokenizer, tokenizerModelType);
    }

    /**
     * Names of every tensor present in the checkpoint, as read from the GGUF tensor info section.
     * <p>
     * The live list is returned. It is a convenience view for diagnostics and for deciding which
     * optional tensors a loader should expect.
     */
    public List<String> getTensorNames() {
        return tensorNames;
    }

    public Config setTensorNames(final List<String> tensorNames) {
        this.tensorNames = tensorNames;
        return this;
    }

    /**
     * {@code <arch>.rope.scaling.original_context_length} - the context length the rope scaling was
     * calibrated for, i.e. the length the scaling factor interpolates from.
     * <p>
     * 0 when the key is absent, meaning no calibrated scaling is available.
     */
    public int getRopeScalingOriginalContextLength() {
        return ropeScalingOriginalContextLength;
    }

    public Config setRopeScalingOriginalContextLength(final int ropeScalingOriginalContextLength) {
        this.ropeScalingOriginalContextLength = ropeScalingOriginalContextLength;
        return this;
    }

    /**
     * {@code <arch>.rope.scaling.attn_factor} - multiplier applied to attention weights after
     * YaRN-style scaling, used to keep attention sharpness stable at scaled positions.
     * <p>
     * 0 when the key is absent, in which case the rope implementation applies no extra factor.
     */
    public float getRopeScalingAttnFactor() {
        return ropeScalingAttnFactor;
    }

    public Config setRopeScalingAttnFactor(final float ropeScalingAttnFactor) {
        this.ropeScalingAttnFactor = ropeScalingAttnFactor;
        return this;
    }

    /**
     * {@code <arch>.rope.dimension_count} - how many dimensions of a head are rotated, counting from
     * the start of the head.
     * <p>
     * 0 (key absent) means rotate the whole head.
     */
    public int getRopeDimCount() {
        return ropeDimCount;
    }

    public Config setRopeDimCount(final int ropeDimCount) {
        this.ropeDimCount = ropeDimCount;
        return this;
    }

    public void setEmbeddingLengthPerLayerInput(final Integer layerEmbeddingLength) {
        this.embeddingLengthPerLayerInput = layerEmbeddingLength;
    }

    /**
     * {@code <arch>.embedding_length_per_layer_input} - width of the per-layer input embeddings some
     * architectures add to the residual stream at each block.
     * <p>
     * 0 when the key is absent, meaning the model has no per-layer input embeddings and the feature
     * is not applied.
     */
    public int getEmbeddingLengthPerLayerInput() {
        return embeddingLengthPerLayerInput;
    }

    /**
     * Internal {@link ChatRole} to template role name mapping, e.g. {@code MODEL -> "assistant"},
     * used when rendering the chat template.
     * <p>
     * Lazily defaults to the Qwen-family mapping when no value was set, and that default is then
     * cached in the config. The live map is returned, so callers must not mutate it.
     */
    public Map<ChatRole, String> getRoleMapping() {
        if (roleMapping == null) {
            // Qwen family role mapping
            roleMapping = Map.of(
                    ChatRole.SYSTEM, "system",
                    ChatRole.USER, "user",
                    ChatRole.MODEL, "assistant"
            );
        }
        return roleMapping;
    }

    public Config setRoleMapping(final Map<ChatRole, String> roleMapping) {
        this.roleMapping = roleMapping;
        return this;
    }

    /**
     * {@code <arch>.attention.key_length} — the per-head key width. Deliberately NOT wired into
     * {@link #getHeadSize()}: headSize is assigned per-model by the checkpoint loaders, and parsing
     * key_length globally would silently change Qwen2/Phi3 sizing.
     */
    public int getAttentionKeyLength() {
        return attentionKeyLength;
    }

    public Config setAttentionKeyLength(final int attentionKeyLength) {
        this.attentionKeyLength = attentionKeyLength;
        return this;
    }

    /**
     * {@code <arch>.full_attention_interval} - in a hybrid model, every Nth layer (1-based) is a
     * full-attention layer and the rest are recurrent (Gated Delta Net) layers.
     * <p>
     * 0 when the key is absent. A stated {@code <arch>.attention.recurrent_layers} list overrides
     * this entirely - see {@link #isFullAttentionLayer(int)}.
     */
    public int getFullAttentionInterval() {
        return fullAttentionInterval;
    }

    public Config setFullAttentionInterval(final int fullAttentionInterval) {
        this.fullAttentionInterval = fullAttentionInterval;
        return this;
    }

    /**
     * Per-layer layout as stated by {@code <arch>.attention.recurrent_layers}: {@code true} means the
     * layer is a recurrent (Gated Delta Net) layer, {@code false} a full-attention layer. Null when the
     * GGUF did not state it, in which case {@link #fullAttentionInterval} is the only clue.
     */
    public boolean[] getRecurrentLayers() {
        return recurrentLayers == null ? null : recurrentLayers.clone();
    }

    public Config setRecurrentLayers(final boolean[] recurrentLayers) {
        this.recurrentLayers = recurrentLayers == null ? null : recurrentLayers.clone();
        return this;
    }

    /**
     * True when the checkpoint stated {@code <arch>.attention.recurrent_layers}, i.e. the layer
     * layout is authoritative rather than derived from {@link #getFullAttentionInterval()}.
     */
    public boolean hasExplicitRecurrentLayers() {
        return recurrentLayers != null;
    }

    /**
     * {@code <arch>.ssm.conv_kernel} - width of the short causal convolution applied to the GDN qkv
     * projection before the recurrence.
     */
    public int getConvKernel() {
        return convKernel;
    }

    public Config setConvKernel(final int convKernel) {
        this.convKernel = convKernel;
        return this;
    }

    /**
     * {@code <arch>.ssm.state_size} - per-head state dimension of the Gated Delta Net recurrence,
     * which must match {@link #getSsmHeadValueDim()}.
     */
    public int getStateSize() {
        return stateSize;
    }

    public Config setStateSize(final int stateSize) {
        this.stateSize = stateSize;
        return this;
    }

    /**
     * {@code <arch>.ssm.group_count} - number of GDN head groups; each group contributes
     * {@code stateSize} to the q and k halves of the fused qkv projection.
     */
    public int getGroupCount() {
        return groupCount;
    }

    public Config setGroupCount(final int groupCount) {
        this.groupCount = groupCount;
        return this;
    }

    /**
     * {@code <arch>.ssm.time_step_rank} - rank of the time-step (delta-rule) projection, and the
     * divisor that turns {@link #getInnerSize()} into the per-head value width
     * ({@link #getSsmHeadValueDim()}).
     */
    public int getTimeStepRank() {
        return timeStepRank;
    }

    public Config setTimeStepRank(final int timeStepRank) {
        this.timeStepRank = timeStepRank;
        return this;
    }

    /**
     * {@code <arch>.ssm.inner_size} - total width of the value half of the fused GDN projection,
     * i.e. {@code timeStepRank * ssmHeadValueDim}.
     */
    public int getInnerSize() {
        return innerSize;
    }

    public Config setInnerSize(final int innerSize) {
        this.innerSize = innerSize;
        return this;
    }

    /**
     * Total width of the fused GDN qkv projection: q and k each contribute
     * {@code groupCount * stateSize}, v contributes {@code timeStepRank * headValueDim}
     * (which equals {@code innerSize} when {@code innerSize / timeStepRank == stateSize}).
     */
    public int getConvChannels() {
        return innerSize + 2 * groupCount * stateSize;
    }

    /**
     * Per-head value/state width of the GDN recurrence: {@code inner_size / time_step_rank}.
     */
    public int getSsmHeadValueDim() {
        return innerSize / timeStepRank;
    }

    /**
     * Width of a gated query projection: per head {@code [headSize query | headSize gate]}.
     */
    public int getGatedQueryWidth() {
        return 2 * getQueryAttentionWidth();
    }

    /**
     * True when the architecture interleaves SSM (linear-attention) layers with full-attention layers.
     * A stated {@code <arch>.attention.recurrent_layers} list decides it; otherwise the presence of
     * {@code <arch>.full_attention_interval} is the clue.
     */
    public boolean isHybridAttention() {
        if (recurrentLayers != null) {
            for (final boolean recurrent : recurrentLayers) {
                if (recurrent) {
                    return true;
                }
            }
            return false;
        }
        return fullAttentionInterval > 0;
    }

    /**
     * Which branch a layer takes: full attention or Gated Delta Net.
     * <p>
     * {@code <arch>.attention.recurrent_layers} is authoritative when the checkpoint states it - a
     * model can end with a full-attention layer that no interval can express. Otherwise full-attention
     * layers are every {@code fullAttentionInterval}-th layer (1-based). When the metadata key is
     * absent (interval 0) every layer is full attention - the model is not hybrid, and modulo by zero
     * would throw.
     */
    public boolean isFullAttentionLayer(final int layer) {
        if (recurrentLayers != null) {
            if (layer < 0 || layer >= recurrentLayers.length) {
                throw new IndexOutOfBoundsException(
                        "attention.recurrent_layers has %d entries, asked for layer %d"
                                .formatted(recurrentLayers.length, layer));
            }
            return !recurrentLayers[layer];
        }
        if (fullAttentionInterval <= 0) {
            return true;
        }
        return (layer + 1) % fullAttentionInterval == 0;
    }
}
