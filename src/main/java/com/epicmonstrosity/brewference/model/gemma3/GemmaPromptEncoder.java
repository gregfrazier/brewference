package com.epicmonstrosity.brewference.model.gemma3;

import com.epicmonstrosity.brewference.tokenizer.encoder.AbstractSentencePiecePromptEncoder;

public class GemmaPromptEncoder extends AbstractSentencePiecePromptEncoder {
    @Override
    protected boolean startsWithBoundaryWhitespace(final String text) {
        return text.startsWith(" ") || text.startsWith("\n") || text.startsWith("\t");
    }

    @Override
    protected boolean shouldAddLeadingBoundary(final int segmentStart) {
        return segmentStart == 0;
    }

    @Override
    protected int unknownTokenId() {
        return 3;
    }
}
