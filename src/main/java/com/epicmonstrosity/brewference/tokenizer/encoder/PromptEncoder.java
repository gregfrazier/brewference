package com.epicmonstrosity.brewference.tokenizer.encoder;

import com.epicmonstrosity.brewference.tokenizer.vocab.Vocabulary;

import java.util.*;

public interface PromptEncoder {
    String GPT2_PATTERN = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\\d+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+";
    String SMAUG_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    String QWEN2_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    /**
     * Builds a mapping table that converts byte values (0-255) to their corresponding
     * Unicode characters used by the tokenizer.
     * <p>
     * This method maps printable ASCII characters and specific Latin-1 ranges
     * directly to themselves, while mapping all other bytes to higher Unicode
     * codepoints to avoid conflicts with control characters or standard text.
     * </p>
     *
     * @return a char array of length 256 where the index represents the byte value
     *         and the value is the mapped Unicode character.
     */
    default char[] buildByteToUnicode() {
        final List<Integer> printableBytes = new ArrayList<>();

        // Define ranges for direct mapping (printable ASCII and specific Latin-1 characters)
        for (int i = '!'; i <= '~'; i++)
            printableBytes.add(i);
        for (int i = 0xA1; i <= 0xAC; i++)
            printableBytes.add(i);
        for (int i = 0xAE; i <= 0xFF; i++)
            printableBytes.add(i);

        final Set<Integer> printableSet = new HashSet<>(printableBytes);
        final List<Integer> unicodeCodepoints = new ArrayList<>(printableBytes);

        // For all bytes not in the printable range, map them to a unique high-range Unicode codepoint
        int nonPrintableCounter = 0;
        for (int b = 0; b < 256; b++) {
            if (!printableSet.contains(b)) {
                printableBytes.add(b);
                unicodeCodepoints.add(256 + nonPrintableCounter);
                nonPrintableCounter++;
            }
        }

        // Construct the lookup table where index = byte value and value = mapped Unicode char
        final char[] table = new char[256];
        for (int i = 0; i < printableBytes.size(); i++) {
            table[printableBytes.get(i)] = (char) (int) unicodeCodepoints.get(i);
        }

        return table;
    }

    default List<Integer> processPrompt(final Vocabulary vocab, final Map<String, Float> scores, final String text) {
        return processPrompt(vocab, text);
    }

    List<Integer> processPrompt(Vocabulary vocab, String text);
}
