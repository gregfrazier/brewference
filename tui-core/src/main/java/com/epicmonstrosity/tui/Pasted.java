package com.epicmonstrosity.tui;

/** Bracketed paste from the terminal, normalized to `\n` line endings. */
public record Pasted(String text) implements UiMessage {
    public Pasted {
        text = normalize(text);
    }

    /**
     * Normalize raw paste text to use only \n line endings.
     *
     * @param raw The raw paste text
     * @return The normalized text with \n line endings, or empty string if null/empty
     */
    public static String normalize(final String raw) {
        if (raw == null || raw.isEmpty()) {
            return "";
        }
        return raw.replace("\r\n", "\n").replace('\r', '\n');
    }
}
