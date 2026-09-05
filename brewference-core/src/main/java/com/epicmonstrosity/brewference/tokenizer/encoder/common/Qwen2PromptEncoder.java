package com.epicmonstrosity.brewference.tokenizer.encoder.common;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

public class Qwen2PromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return QWEN2_PATTERN;
    }
}
