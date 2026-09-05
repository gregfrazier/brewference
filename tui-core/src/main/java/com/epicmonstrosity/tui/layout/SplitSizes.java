package com.epicmonstrosity.tui.layout;

/**
 * Pixel-like cell counts for a two-pane split.
 * {@code first + second + divider == total}.
 */
public record SplitSizes(int first, int second, int divider) {
    /**
     * Get the total cells.
     *
     * @return The total
     */
    public int total() {
        return first + second + divider;
    }
}
