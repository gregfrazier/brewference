package com.epicmonstrosity.brewference.model.phi3;

import com.epicmonstrosity.brewference.tokenizer.encoder.AbstractSentencePiecePromptEncoder;

public class Phi3PromptEncoder extends AbstractSentencePiecePromptEncoder {
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
