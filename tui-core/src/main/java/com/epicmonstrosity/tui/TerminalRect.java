package com.epicmonstrosity.tui;

/**
 * Interface representing the terminal dimensions.
 * Implemented by {@link TerminalRenderer}.
 */
public interface TerminalRect {
    /**
     * Get the current terminal width in columns.
     *
     * @return The terminal width
     */
    int getWidth();

    /**
     * Get the current terminal height in rows.
     *
     * @return The terminal height
     */
    int getHeight();
}
