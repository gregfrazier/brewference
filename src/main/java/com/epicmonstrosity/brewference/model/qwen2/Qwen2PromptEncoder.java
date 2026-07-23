package com.epicmonstrosity.brewference.model.qwen2;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

public class Qwen2PromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return QWEN2_PATTERN;
    }
}
