package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.ansi.Theme;
import com.epicmonstrosity.tui.layout.TextBlocks;

import java.util.ArrayList;
import java.util.List;

/**
 * Paints a non-modal status box over the bottom of the body.
 * Header and sticky footer stay put; keys still go to the screen.
 */
public final class ToastBanner {
    private static final int MAX_MESSAGE_LINES = 3;

    private ToastBanner() {
    }

    /**
     * Paint a single toast.
     *
     * @param body The underlying body
     * @param toast The toast
     * @param cols The terminal columns
     * @param bodyRows The body rows
     * @return The painted toast
     */
    public static String paint(final String body, final Toast toast, final int cols, final int bodyRows) {
        return paint(body, toast == null ? List.of() : List.of(toast), null, cols, bodyRows, 0);
    }

    /**
     * Paint a toast with pending count.
     *
     * @param body The underlying body
     * @param toast The toast
     * @param cols The terminal columns
     * @param bodyRows The body rows
     * @param pending The pending toast count
     * @return The painted toast
     */
    public static String paint(final String body,
                               final Toast toast,
                               final int cols,
                               final int bodyRows,
                               final int pending) {
        return paint(body, toast == null ? List.of() : List.of(toast), null, cols, bodyRows, pending);
    }

    /**
     * Paint multiple toasts and an optional status.
     *
     * @param body The underlying body
     * @param toasts The toasts to paint
     * @param status The status toast
     * @param cols The terminal columns
     * @param bodyRows The body rows
     * @return The painted toasts
     */
    public static String paint(final String body,
                               final List<Toast> toasts,
                               final Toast status,
                               final int cols,
                               final int bodyRows) {
        return paint(body, toasts, status, cols, bodyRows, 0);
    }

    /**
     * Paint multiple toasts and an optional status with pending count.
     *
     * @param body The underlying body
     * @param toasts The toasts to paint
     * @param status The status toast
     * @param cols The terminal columns
     * @param bodyRows The body rows
     * @param pending The pending toast count
     * @return The painted toasts
     */
    public static String paint(final String body,
                               final List<Toast> toasts,
                               final Toast status,
                               final int cols,
                               final int bodyRows,
                               final int pending) {
        if (bodyRows <= 0) {
            return body == null ? "" : body;
        }
        final List<List<String>> boxes = new ArrayList<>();
        boolean firstToast = true;
        if (toasts != null) {
            for (Toast toast : toasts) {
                if (toast == null || toast.expired()) {
                    continue;
                }
                boxes.add(renderBox(toast, cols, firstToast ? pending : 0));
                firstToast = false;
            }
        }
        if (status != null && !status.expired()) {
            boxes.add(renderBox(status, cols, 0));
        }
        if (boxes.isEmpty()) {
            return body == null ? "" : body;
        }

        final List<String> lines = new ArrayList<>(TextBlocks.lines(body));
        while (lines.size() < bodyRows) {
            lines.add("");
        }
        final List<String> clipped = lines.size() > bodyRows
                ? new ArrayList<>(lines.subList(0, bodyRows))
                : lines;

        final int gap = bodyRows > 1 ? 1 : 0;
        final int budget = Math.max(1, bodyRows - gap);
        while (heightOf(boxes) > budget && boxes.size() > 1) {
            boxes.removeFirst();
        }
        final List<String> stack = flatten(boxes);
        final List<String> fitted = stack.size() > budget
                ? new ArrayList<>(stack.subList(stack.size() - budget, stack.size()))
                : stack;

        final int top = bodyRows - gap - fitted.size();
        final StringBuilder out = new StringBuilder();
        for (int r = 0; r < bodyRows; r++) {
            if (r > 0) {
                out.append('\n');
            }
            final int stackIndex = r - top;
            if (stackIndex >= 0 && stackIndex < fitted.size()) {
                final String boxLine = fitted.get(stackIndex);
                final int boxWidth = TextBlocks.visibleLength(boxLine);
                final int left = Math.max(0, (cols - boxWidth) / 2);
                out.append(" ".repeat(left)).append(boxLine);
            } else {
                out.append(clipped.get(r));
            }
        }
        return out.toString();
    }

    /**
     * Get the total height of toast boxes.
     *
     * @param boxes The toast boxes
     * @return The total height
     */
    private static int heightOf(final List<List<String>> boxes) {
        int height = 0;
        for (List<String> box : boxes) {
            height += box.size();
        }
        return height;
    }

    /**
     * Flatten a list of toast boxes.
     *
     * @param boxes The toast boxes
     * @return The flattened list
     */
    private static List<String> flatten(final List<List<String>> boxes) {
        final List<String> stack = new ArrayList<>();
        for (List<String> box : boxes) {
            stack.addAll(box);
        }
        return stack;
    }

    /**
     * Render a toast as a single line.
     *
     * @param toast The toast
     * @param cols The terminal columns
     * @return The rendered toast
     */
    static String render(final Toast toast, final int cols) {
        return renderBox(toast, cols, 0).get(1);
    }

    /**
     * Render a toast box.
     *
     * @param toast The toast
     * @param cols The terminal columns
     * @param pending The pending count
     * @return The rendered box lines
     */
    private static List<String> renderBox(final Toast toast, final int cols, final int pending) {
        final Theme theme = Theme.current();
        final String style = toast.style();
        final String label = switch (toast.kind()) {
            case SUCCESS -> "ok";
            case WARNING -> "warn";
            case ERROR -> "err";
            case INFO -> "info";
        };
        final String title = pending > 0
                ? " " + label + " +" + pending + " "
                : " " + label + " ";

        final int maxInner = Math.max(8, cols - 6);
        List<String> wrapped = TextBlocks.wrapVisible(toast.message() == null ? "" : toast.message(), maxInner);
        if (wrapped.size() > MAX_MESSAGE_LINES) {
            wrapped = new ArrayList<>(wrapped.subList(0, MAX_MESSAGE_LINES));
        }
        int inner = Math.max(8, TextBlocks.visibleLength(title));
        for (String line : wrapped) {
            inner = Math.max(inner, TextBlocks.visibleLength(line));
        }
        inner = Math.min(maxInner, inner);
        final int boxWidth = Math.min(Math.max(1, cols), inner + 4);

        final List<String> box = new ArrayList<>();
        box.add(horizontal('┌', '┐', title, boxWidth, theme, style));
        for (String line : wrapped) {
            box.add(edge(theme, style)
                    + "│ "
                    + theme.reset()
                    + style
                    + TextBlocks.fit(line, Math.max(1, boxWidth - 4))
                    + edge(theme, style)
                    + " │"
                    + theme.reset());
        }
        box.add(horizontal('└', '┘', "", boxWidth, theme, style));
        return box;
    }

    /**
     * Create a horizontal box line.
     *
     * @param left The left corner character
     * @param right The right corner character
     * @param title The title
     * @param width The width
     * @param theme The color theme
     * @param style The style code
     * @return The horizontal line
     */
    private static String horizontal(final char left,
                                     final char right,
                                     final String title,
                                     final int width,
                                     final Theme theme,
                                     final String style) {
        final String prefix = left + title;
        final int fill = Math.max(0, width - TextBlocks.visibleLength(prefix) - 1);
        return edge(theme, style) + prefix + "─".repeat(fill) + right + theme.reset();
    }

    /**
     * Get the edge styling.
     *
     * @param theme The color theme
     * @param style The style code
     * @return The edge string
     */
    private static String edge(final Theme theme, final String style) {
        return theme.bold() + style;
    }
}
