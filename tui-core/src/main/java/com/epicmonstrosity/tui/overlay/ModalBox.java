package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.ansi.Theme;
import com.epicmonstrosity.tui.layout.TextBlocks;

import java.util.ArrayList;
import java.util.List;

/**
 * Paints a centered modal box over the underlying screen body, dimming the
 * uncovered background. {@link #paint} auto-fits the box to its content;
 * {@link #paintSized} paints a fixed-size box (percent of the terminal),
 * padding or truncating the content — useful for modals with scrolling
 * content such as a file picker.
 */
public final class ModalBox {
    private static final int MIN_BOX_WIDTH = 20;
    private static final int MIN_BOX_HEIGHT = 6;

    private ModalBox() {
    }

    /** Fixed box dimensions with derived inner (content) dimensions. */
    public record Size(int boxWidth, int boxHeight) {
        public int innerWidth() {
            return Math.max(2, boxWidth - 4);
        }

        public int innerHeight() {
            return Math.max(1, boxHeight - 2);
        }
    }

    /**
     * Compute a box size as a percentage of the terminal, clamped to sane
     * minimums and to the available space.
     */
    public static Size size(final ViewContext ctx, final int widthPercent, final int heightPercent) {
        final int cols = ctx.terminalCols();
        final int bodyRows = ctx.bodyRows();
        final int boxWidth = Math.min(cols - 2, Math.max(MIN_BOX_WIDTH, cols * clampPct(widthPercent) / 100));
        final int boxHeight = Math.min(bodyRows, Math.max(MIN_BOX_HEIGHT, bodyRows * clampPct(heightPercent) / 100));
        return new Size(Math.max(4, boxWidth), Math.max(3, boxHeight));
    }

    /**
     * Paint a modal box over the underlying body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @param title The modal title
     * @param rows The content rows
     * @return The painted modal box
     */
    static String paint(final ViewContext ctx, final String underlyingBody, final String title, final List<String> rows) {
        final int cols = ctx.terminalCols();
        final int bodyRows = ctx.bodyRows();

        int innerWidth = 8;
        for (String row : rows) {
            innerWidth = Math.max(innerWidth, visibleLength(row));
        }
        innerWidth = Math.min(Math.max(cols - 6, 10), innerWidth);
        final List<String> wrapped = wrapRows(rows, innerWidth);
        final int boxWidth = Math.min(cols - 2, innerWidth + 4);
        final int boxHeight = Math.min(bodyRows, wrapped.size() + 2);
        return compose(ctx, underlyingBody, title, rows, new Size(boxWidth, boxHeight));
    }

    /** Paint a fixed-size modal box; content is padded with blanks or truncated to fit. */
    public static String paintSized(final ViewContext ctx,
                                    final String underlyingBody,
                                    final String title,
                                    final List<String> rows,
                                    final Size size) {
        return compose(ctx, underlyingBody, title, rows, size);
    }

    private static String compose(final ViewContext ctx,
                                  final String underlyingBody,
                                  final String title,
                                  final List<String> rows,
                                  final Size size) {
        final int cols = ctx.terminalCols();
        final int bodyRows = ctx.bodyRows();
        final String[] under = underlyingBody == null ? new String[0] : underlyingBody.split("\n", -1);

        final int innerWidth = size.innerWidth();
        final int innerHeight = size.innerHeight();
        final List<String> wrapped = wrapRows(rows, innerWidth);
        final int boxWidth = size.boxWidth();
        final int boxHeight = size.boxHeight();
        final int top = Math.max(0, (bodyRows - boxHeight) / 2);
        final int left = Math.max(0, (cols - boxWidth) / 2);

        final List<String> box = new ArrayList<>();
        box.add(horizontal('┌', '┐', "─ " + title + " ", boxWidth));
        for (int i = 0; i < innerHeight; i++) {
            final String text = i < wrapped.size() ? wrapped.get(i) : "";
            final Theme theme = Theme.current();
            box.add(theme.bold() + "│ " + theme.reset() + pad(text, boxWidth - 4) + theme.bold() + " │" + theme.reset());
        }
        box.add(horizontal('└', '┘', "", boxWidth));

        final StringBuilder out = new StringBuilder();
        for (int r = 0; r < bodyRows; r++) {
            if (r > 0) {
                out.append('\n');
            }
            if (r >= top && r < top + box.size()) {
                out.append(" ".repeat(left)).append(box.get(r - top));
            } else if (r < under.length) {
                final Theme theme = Theme.current();
                out.append(theme.dim()).append(under[r]).append(theme.reset());
            }
        }
        return out.toString();
    }

    private static int clampPct(final int percent) {
        return Math.max(0, Math.min(100, percent));
    }

    /**
     * Get the visible length of text.
     *
     * @param text The text
     * @return The visible length
     */
    static int visibleLength(final String text) {
        return TextBlocks.visibleLength(text);
    }

    /**
     * Pad text to a width.
     *
     * @param text The text
     * @param width The target width
     * @return The padded text
     */
    static String pad(final String text, final int width) {
        return TextBlocks.fit(text, width);
    }

    /**
     * Wrap rows to a width.
     *
     * @param rows The rows to wrap
     * @param width The target width
     * @return The wrapped rows
     */
    private static List<String> wrapRows(final List<String> rows, final int width) {
        final List<String> wrapped = new ArrayList<>();
        for (String row : rows) {
            if (row == null || row.isEmpty()) {
                wrapped.add("");
                continue;
            }
            wrapped.addAll(TextBlocks.wrapVisible(row, width));
        }
        return wrapped;
    }

    /**
     * Create a horizontal box line.
     *
     * @param left The left corner character
     * @param right The right corner character
     * @param title The title
     * @param width The width
     * @return The horizontal line
     */
    private static String horizontal(final char left, final char right, final String title, final int width) {
        final Theme theme = Theme.current();
        final String prefix = left + title;
        final int fill = Math.max(0, width - visibleLength(prefix) - 1);
        return theme.bold() + prefix + "─".repeat(fill) + right + theme.reset();
    }
}
