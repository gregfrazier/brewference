package com.epicmonstrosity.brewference.tokenizer.encoder;

import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class BytePairPromptEncoder implements PromptEncoder {
    private final Pattern splitPattern = Pattern.compile(splitPattern(), Pattern.UNICODE_CHARACTER_CLASS);

    private final char[] byteToUnicode = this.buildByteToUnicode();
    private final Map<String, List<String>> bpeCache = new HashMap<>();

    protected abstract String splitPattern();

    @Override
    public List<Integer> processPrompt(final Vocabulary vocab, final String text) {
        final List<Integer> result = new ArrayList<>();
        if (vocab.getSpecialTokensPattern() == null) {
            return tokenizeText(vocab, text, true);
        }
        final Matcher matcher = vocab.getSpecialTokensPattern().matcher(text);
        int segmentStart = 0;
        while(matcher.find()) {
            if (matcher.start() > segmentStart) {
                result.addAll(tokenizeText(vocab, text.substring(segmentStart, matcher.start()), segmentStart == 0));
            }
            result.add(vocab.getTokenToId().get(matcher.group()));
            segmentStart = matcher.end();
        }
        if (segmentStart < text.length()) {
            result.addAll(tokenizeText(vocab, text.substring(segmentStart), segmentStart == 0));
        }

        return result;
    }

    private List<Integer> tokenizeText(final Vocabulary vocab, final String text, final boolean isFirstSegment) {
        final List<Integer> result = new ArrayList<>();
        if (text.isEmpty()) {
            return result;
        }
        final String normalized = (vocab.getAddPrefixSpace() && isFirstSegment && text.charAt(0) != ' ') ? " " + text : text;
        final Matcher splitMatcher = splitPattern.matcher(normalized);
        while (splitMatcher.find()) {
            final String found = toByteLevel(splitMatcher.group());
            for (final String sub : applyBytePairEncoding(vocab, found)) {
                final Integer tokenId = vocab.getTokenToId().get(sub);
                if (tokenId != null) {
                    result.add(tokenId);
                } else {
                    throw new IllegalStateException("Cannot find tokenId; Byte-level fallback not supported in GPT2 BPE");
                }
            }
        }

        return result;
    }

    private String toByteLevel(final String piece) {
        final byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        final StringBuilder encodedStringBuilder = new StringBuilder(bytes.length);
        for (final byte byteValue : bytes) {
            encodedStringBuilder.append(byteToUnicode[byteValue & 0xFF]);
        }
        return encodedStringBuilder.toString();
    }

    private List<String> applyBytePairEncoding(final Vocabulary vocab, final String piece) {
        final List<String> cached = bpeCache.get(piece);
        if (cached != null) return cached;

        List<String> word = new ArrayList<>(piece.length());
        for (int i = 0; i < piece.length(); i++) {
            word.add(String.valueOf(piece.charAt(i)));
        }

        while (word.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                final Integer r = vocab.getMergeRanks().get(word.get(i) + '\u0000' + word.get(i + 1));
                if (r != null && r < bestRank) {
                    bestRank = r;
                    bestIdx = i;
                }
            }
            if (bestIdx == -1) break;

            final List<String> merged = new ArrayList<>(word.size() - 1);
            int i = 0;
            while (i < word.size()) {
                if (i == bestIdx) {
                    merged.add(word.get(i) + word.get(i + 1));
                    i += 2;
                } else {
                    merged.add(word.get(i));
                    i++;
                }
            }
            word = merged;
        }

        bpeCache.put(piece, word);
        return word;
    }
}
