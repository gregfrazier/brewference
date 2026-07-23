package com.epicmonstrosity.brewference.model.llama2;

import com.epicmonstrosity.brewference.tokenizer.encoder.AbstractSentencePiecePromptEncoder;

public class LlamaPromptEncoder extends AbstractSentencePiecePromptEncoder {
    @Override
    protected boolean startsWithBoundaryWhitespace(final String text) {
        return text.charAt(0) == ' ';
    }

    @Override
    protected boolean shouldAddLeadingBoundary(final int segmentStart) {
        return true;
    }

    @Override
    protected int unknownTokenId() {
        return 0;
    }
}
