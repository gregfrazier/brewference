package com.epicmonstrosity.brewference.cli.screen.chat;

/**
 * A single line of wrapped chat message content.
 */
public record ChatLine(String role, String gutter, String text, boolean continuation) {
    /** Separator between gutter and text. */
    public static final String SEPARATOR = " │ ";

    /**
     * Get the formatted line with gutter and separator.
     *
     * @return The formatted line
     */
    public String formatted() {
        return gutter + SEPARATOR + (text == null ? "" : text);
    }
}
