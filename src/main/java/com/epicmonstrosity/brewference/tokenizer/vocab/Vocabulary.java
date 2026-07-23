package com.epicmonstrosity.brewference.tokenizer.vocab;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class Vocabulary {
    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final List<String> specialTokens = new ArrayList<>();
    private final Map<String, Float> tokenToScore = new HashMap<>();
    private final List<String> merges = new ArrayList<>();
    private final Map<String, Integer> mergeRanks = new HashMap<>();
    private Pattern specialTokensPattern = null;
    private Boolean addPrefixSpace = false;

    public Vocabulary addTokenId(final String token, final Integer id) {
        this.tokenToId.put(token, id);
        this.idToToken.put(id, token);
        return this;
    }

    public List<String> getSpecialTokens() {
        return specialTokens;
    }

    public Vocabulary addSpecialTokens(final String specialToken) {
        this.specialTokens.add(specialToken);
        return this;
    }

    public Vocabulary addTokenScore(final String token, final float score) {
        this.tokenToScore.put(token, score);
        return this;
    }

    public Vocabulary addTokenMerge(final String token) {
        this.merges.add(token);
        return this;
    }

    public List<String> getMerges() {
        return merges;
    }

    public Map<String, Integer> getTokenToId() {
        return tokenToId;
    }

    public Map<Integer, String> getIdToToken() {
        return idToToken;
    }

    public Map<String, Float> getTokenToScore() {
        return tokenToScore;
    }

    public Boolean getAddPrefixSpace() {
        return addPrefixSpace;
    }

    public Vocabulary setAddPrefixSpace(final Boolean addPrefixSpace) {
        this.addPrefixSpace = addPrefixSpace;
        return this;
    }

    public Map<String, Integer> getMergeRanks() {
        if (mergeRanks.isEmpty()) {
            for (int i = 0; i < merges.size(); i++) {
                final String merge = extractMerge(i);
                mergeRanks.put(merge.replaceFirst(" ", "\u0000"), i);
            }
        }
        return mergeRanks;
    }

    private String extractMerge(final int index) {
        final String merge = merges.get(index);
        final int spaceIdx = merge.indexOf(' ');
        if (spaceIdx < 0)
            throw new IllegalArgumentException("Malformed merge entry at index " + index + ": \"" + merge + "\"");
        return merge;
    }

    public Pattern getSpecialTokensPattern() {
        if (specialTokensPattern == null && !specialTokens.isEmpty()) {
            final List<String> specialTokensSorted = new ArrayList<>(this.specialTokens);
            specialTokensSorted.sort((o1, o2) -> o2.length() - o1.length());
            final StringBuilder sb = new StringBuilder();
            for (final String token : specialTokensSorted) {
                if (sb.length() > 0)
                    sb.append('|');
                sb.append(Pattern.quote(token));
            }
            specialTokensPattern = Pattern.compile(sb.toString());
        }
        return specialTokensPattern;
    }
}
