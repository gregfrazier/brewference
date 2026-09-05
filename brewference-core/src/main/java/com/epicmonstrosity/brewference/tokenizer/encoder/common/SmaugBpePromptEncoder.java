package com.epicmonstrosity.brewference.tokenizer.encoder.common;

import com.epicmonstrosity.brewference.tokenizer.encoder.BytePairPromptEncoder;

// Smaug and Databricks style
public class SmaugBpePromptEncoder extends BytePairPromptEncoder {
    @Override
    protected String splitPattern() {
        return SMAUG_PATTERN;
    }
}
