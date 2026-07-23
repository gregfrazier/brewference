package com.epicmonstrosity.brewference.tokenizer.encoder;

import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public abstract class AbstractSentencePiecePromptEncoder implements PromptEncoder {
    protected static final String SPACE_MARKER = "▁";

    @Override
    public List<Integer> processPrompt(final Vocabulary vocab, final String text) {
        final Map<String, Float> scores = vocab.getTokenToScore();
        final List<Integer> result = new ArrayList<>();

        int segmentStart = 0;

        while (segmentStart < text.length()) {
            final SpecialTokenMatch specialToken = findNextSpecialToken(vocab, text, segmentStart);

            if (specialToken == null) {
                result.addAll(tokenizeTextSegment(vocab, scores, text.substring(segmentStart), shouldAddLeadingBoundary(segmentStart)));
                break;
            }

            if (specialToken.start > segmentStart) {
                result.addAll(tokenizeTextSegment(vocab, scores, text.substring(segmentStart, specialToken.start), shouldAddLeadingBoundary(segmentStart)));
            }

            result.add(specialToken.tokenId);
            segmentStart = specialToken.end;
        }

        return result;
    }

    protected List<Integer> tokenizeTextSegment(final Vocabulary vocab, final Map<String, Float> scores, final String text, final boolean addLeadingBoundary) {
        final List<Integer> result = new ArrayList<>();

        if (text.isEmpty()) {
            return result;
        }

        final String normalized = normalizeForSentencePiece(text, addLeadingBoundary);
        final List<String> pieces = splitCodePoints(normalized);

        mergeBestScoringPairs(pieces, scores);

        for (final String piece : pieces) {
            final Integer tokenId = vocab.getTokenToId().get(piece);
            if (tokenId != null) {
                result.add(tokenId);
            } else {
                appendByteFallback(vocab, result, piece);
            }
        }

        return result;
    }

    protected String normalizeForSentencePiece(final String text, final boolean addLeadingBoundary) {
        final String withBoundary = addLeadingBoundary && !startsWithBoundaryWhitespace(text)
                ? SPACE_MARKER + text
                : text;

        return withBoundary.replace(" ", SPACE_MARKER);
    }

    protected List<String> splitCodePoints(final String text) {
        final List<String> pieces = new ArrayList<>();

        for (int offset = 0; offset < text.length();) {
            final int codePoint = text.codePointAt(offset);
            pieces.add(new String(Character.toChars(codePoint)));
            offset += Character.charCount(codePoint);
        }

        return pieces;
    }

    protected void mergeBestScoringPairs(final List<String> pieces, final Map<String, Float> scores) {
        while (pieces.size() > 1) {
            int bestIndex = -1;
            float bestScore = Float.NEGATIVE_INFINITY;

            for (int index = 0; index < pieces.size() - 1; index++) {
                final String merged = pieces.get(index) + pieces.get(index + 1);
                final Float score = scores.get(merged);

                if (score != null && score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }

            if (bestIndex < 0) {
                return;
            }

            pieces.set(bestIndex, pieces.get(bestIndex) + pieces.get(bestIndex + 1));
            pieces.remove(bestIndex + 1);
        }
    }

    protected void appendByteFallback(final Vocabulary vocab, final List<Integer> result, final String piece) {
        for (final byte value : piece.getBytes(StandardCharsets.UTF_8)) {
            final String byteToken = String.format("<0x%02X>", value & 0xFF);
            result.add(vocab.getTokenToId().getOrDefault(byteToken, unknownTokenId()));
        }
    }

    protected abstract boolean startsWithBoundaryWhitespace(String text);

    protected abstract boolean shouldAddLeadingBoundary(int segmentStart);

    protected abstract int unknownTokenId();

    protected SpecialTokenMatch findNextSpecialToken(final Vocabulary vocab, final String text, final int searchFrom) {
        SpecialTokenMatch match = null;

        for (final String specialToken : vocab.getSpecialTokens()) {
            final int start = text.indexOf(specialToken, searchFrom);
            if (start < 0)
                continue;

            if (match == null
                    || start < match.start
                    || (start == match.start
                    && specialToken.length() > match.end - match.start)) {
                final Integer tokenId = vocab.getTokenToId().get(specialToken);
                if (tokenId != null) {
                    match = new SpecialTokenMatch(start, start + specialToken.length(), tokenId);
                }
            }
        }

        return match;
    }

    protected static final class SpecialTokenMatch {
        public final int start;
        public final int end;
        public final int tokenId;

        private SpecialTokenMatch(final int start, final int end, final int tokenId) {
            this.start = start;
            this.end = end;
            this.tokenId = tokenId;
        }
    }
}
