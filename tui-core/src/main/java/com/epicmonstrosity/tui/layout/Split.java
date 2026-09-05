package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.ansi.Ansi;

import java.util.List;

/**
 * Two-pane layout math and string stitching. Not a widget tree:
 * you render each side yourself, then join the blocks.
 *
 * <p>Percent is the first pane's share of the space left after a one-cell
 * divider. {@link #DEFAULT_PERCENT} is 50.
 */
public final class Split {
    /** Default percent for two-pane splits. */
    public static final int DEFAULT_PERCENT = 50;

    private Split() {
    }

    /**
     * Calculate split sizes with default percent.
     *
     * @param total The total cells
     * @return The SplitSizes
     */
    public static SplitSizes sizes(final int total) {
        return sizes(total, DEFAULT_PERCENT);
    }

    /**
     * Calculate split sizes.
     *
     * @param total The total cells
     * @param firstPercent The first pane percentage
     * @return The SplitSizes
     */
    public static SplitSizes sizes(final int total, final int firstPercent) {
        return sizes(total, firstPercent, 1);
    }

    /**
     * Calculate split sizes with divider.
     *
     * @param total The total cells
     * @param firstPercent The first pane percentage
     * @param divider The divider width
     * @return The SplitSizes
     */
    public static SplitSizes sizes(final int total, final int firstPercent, final int divider) {
        final int cells = Math.max(0, total);
        if (cells < 2) {
            return new SplitSizes(cells, 0, 0);
        }
        final int bar = Math.max(0, Math.min(divider, cells - 2));
        final int remaining = cells - bar;
        final int percent = Math.max(1, Math.min(99, firstPercent));
        int first = Math.max(1, remaining * percent / 100);
        if (remaining >= 2) {
            first = Math.min(first, remaining - 1);
        }
        return new SplitSizes(first, remaining - first, bar);
    }

    /**
     * Render two blocks vertically with default percent.
     *
     * @param left The left block
     * @param right The right block
     * @param cols The columns
     * @param rows The rows
     * @return The rendered string
     */
    public static String vertical(final String left, final String right, final int cols, final int rows) {
        return vertical(left, right, cols, rows, DEFAULT_PERCENT);
    }

    /**
     * Render two blocks vertically.
     *
     * @param left The left block
     * @param right The right block
     * @param cols The columns
     * @param rows The rows
     * @param firstPercent The first pane percentage
     * @return The rendered string
     */
    public static String vertical(final String left, final String right,
                                  final int cols, final int rows, final int firstPercent) {
        final SplitSizes sizes = sizes(cols, firstPercent);
        final List<String> leftLines = TextBlocks.fitBlock(left, sizes.first(), rows);
        final List<String> rightLines = TextBlocks.fitBlock(right, sizes.second(), rows);
        final StringBuilder out = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            if (i > 0) {
                out.append('\n');
            }
            out.append(leftLines.get(i));
            if (sizes.divider() > 0) {
                out.append(Ansi.DIM).append('│').append(Ansi.RESET);
            }
            out.append(rightLines.get(i));
        }
        return out.toString();
    }

    /**
     * Render two blocks horizontally with default percent.
     *
     * @param top The top block
     * @param bottom The bottom block
     * @param cols The columns
     * @param rows The rows
     * @return The rendered string
     */
    public static String horizontal(final String top, final String bottom, final int cols, final int rows) {
        return horizontal(top, bottom, cols, rows, DEFAULT_PERCENT);
    }

    /**
     * Render two blocks horizontally.
     *
     * @param top The top block
     * @param bottom The bottom block
     * @param cols The columns
     * @param rows The rows
     * @param firstPercent The first pane percentage
     * @return The rendered string
     */
    public static String horizontal(final String top, final String bottom,
                                    final int cols, final int rows, final int firstPercent) {
        final SplitSizes sizes = sizes(rows, firstPercent);
        final List<String> topLines = TextBlocks.fitBlock(top, cols, sizes.first());
        final List<String> bottomLines = TextBlocks.fitBlock(bottom, cols, sizes.second());
        final StringBuilder out = new StringBuilder();
        appendLines(out, topLines);
        if (sizes.divider() > 0) {
            if (!out.isEmpty()) {
                out.append('\n');
            }
            out.append(Ansi.DIM).append("─".repeat(Math.max(0, cols))).append(Ansi.RESET);
        }
        appendLines(out, bottomLines);
        return out.toString();
    }

    /**
     * Append lines to a StringBuilder.
     *
     * @param out The StringBuilder to append to
     * @param lines The lines to append
     */
    private static void appendLines(final StringBuilder out, final List<String> lines) {
        for (String line : lines) {
            if (!out.isEmpty()) {
                out.append('\n');
            }
            out.append(line);
        }
    }
}
