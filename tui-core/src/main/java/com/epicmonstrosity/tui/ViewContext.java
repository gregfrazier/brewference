package com.epicmonstrosity.tui;

/**
 * Passed from {@link TerminalRenderer} into a screen on every render.
 * Tells the view exactly how many rows are available for the scrollable body
 * so it can compute the correct viewport window.
 */
public record ViewContext(
        int terminalRows,
        int terminalCols,
        int bodyRows        // rows available between header and footer
) {
    /** Clamp a value to [0, max] */
    public static int clamp(final int val, final int max) {
        return Math.max(0, Math.min(val, max));
    }

    /**
     * Compute the scroll offset needed to keep {@code cursor} visible
     * inside a viewport of {@code viewportHeight} rows,
     * given the current {@code currentOffset}.
     */
    public static int scrollOffset(final int cursor, final int currentOffset, final int viewportHeight) {
        if (viewportHeight <= 0)
            return 0;
        if (cursor < currentOffset)
            return cursor;
        if (cursor >= currentOffset + viewportHeight)
            return cursor - viewportHeight + 1;

        return currentOffset;
    }
}
