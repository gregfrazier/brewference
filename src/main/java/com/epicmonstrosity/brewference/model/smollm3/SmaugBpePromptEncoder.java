package com.epicmonstrosity.brewference.model.smollm3;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

public class SmaugBpePromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return SMAUG_PATTERN;
    }
}
