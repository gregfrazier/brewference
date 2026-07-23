package com.epicmonstrosity.brewference.model.smollm;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

public class SmolLMPromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return GPT2_PATTERN;
    }
}
