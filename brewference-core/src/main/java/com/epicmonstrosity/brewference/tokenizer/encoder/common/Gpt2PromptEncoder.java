package com.epicmonstrosity.brewference.tokenizer.encoder.common;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

public class Gpt2PromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return GPT2_PATTERN;
    }
}
