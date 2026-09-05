package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.ansi.DisplayWidth;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * ANSI-aware string helpers for stitching pre-rendered body blocks.
 */
public final class TextBlocks {
    private static final Pattern ANSI = Pattern.compile("\u001B\\[[\\d;]*m");

    private TextBlocks() {
    }

    /**
     * Strip ANSI codes from text.
     *
     * @param text The text
     * @return The text without ANSI codes
     */
    public static String stripAnsi(final String text) {
        return text == null ? "" : ANSI.matcher(text).replaceAll("");
    }

    /**
     * Get the visible length of text (excluding ANSI codes).
     *
     * @param text The text
     * @return The visible length
     */
    public static int visibleLength(final String text) {
        return DisplayWidth.of(stripAnsi(text));
    }

    /**
     * Fit text to a width, padding with spaces if needed.
     *
     * @param text The text
     * @param width The target width
     * @return The fitted text
     */
    public static String fit(final String text, final int width) {
        if (width <= 0) {
            return "";
        }
        final String truncated = truncateVisible(text, width);
        final int visible = visibleLength(truncated);
        if (visible >= width) {
            return truncated;
        }
        return truncated + " ".repeat(width - visible);
    }

    /**
     * Truncate text to visible length.
     *
     * @param text The text
     * @param width The target width
     * @return The truncated text
     */
    public static String truncateVisible(final String text, final int width) {
        if (text == null || width <= 0) {
            return "";
        }
        final StringBuilder out = new StringBuilder();
        int visible = 0;
        for (int i = 0; i < text.length(); ) {
            if (text.charAt(i) == '\u001B') {
                final int end = skipAnsi(text, i);
                out.append(text, i, end);
                i = end;
                continue;
            }
            final int cp = text.codePointAt(i);
            final int n = Character.charCount(cp);
            final int w = DisplayWidth.of(cp);
            if (visible + w > width) {
                break;
            }
            out.appendCodePoint(cp);
            visible += w;
            i += n;
        }
        return out.toString();
    }

    /**
     * Wrap text to a width, respecting ANSI codes.
     *
     * @param text The text
     * @param width The target width
     * @return The wrapped lines
     */
    public static List<String> wrapVisible(final String text, final int width) {
        if (text == null || text.isEmpty()) {
            return List.of("");
        }
        final int cols = Math.max(1, width);
        final List<String> lines = new ArrayList<>();
        final StringBuilder line = new StringBuilder();
        int visible = 0;
        for (int i = 0; i < text.length(); ) {
            if (text.charAt(i) == '\u001B') {
                final int end = skipAnsi(text, i);
                line.append(text, i, end);
                i = end;
                continue;
            }
            final int cp = text.codePointAt(i);
            final int n = Character.charCount(cp);
            final int w = DisplayWidth.of(cp);
            if (w > cols) {
                if (!line.isEmpty()) {
                    lines.add(line.toString());
                    line.setLength(0);
                    visible = 0;
                }
                lines.add(text.substring(i, i + n));
                i += n;
                continue;
            }
            if (visible + w > cols && !line.isEmpty()) {
                lines.add(line.toString());
                line.setLength(0);
                visible = 0;
            }
            line.appendCodePoint(cp);
            visible += w;
            i += n;
        }
        if (!line.isEmpty()) {
            lines.add(line.toString());
        }
        return lines.isEmpty() ? List.of("") : lines;
    }

    /**
     * Split a block into lines.
     *
     * @param block The block
     * @return The lines
     */
    public static List<String> lines(final String block) {
        if (block == null || block.isEmpty()) {
            return List.of();
        }
        final String normalized = block.endsWith("\n") ? block.substring(0, block.length() - 1) : block;
        if (normalized.isEmpty()) {
            return List.of();
        }
        return List.of(normalized.split("\n", -1));
    }

    /**
     * Fit a block to width and height.
     *
     * @param block The block
     * @param width The target width
     * @param height The target height
     * @return The fitted lines
     */
    public static List<String> fitBlock(final String block, final int width, final int height) {
        final List<String> source = lines(block);
        final List<String> out = new ArrayList<>(Math.max(0, height));
        for (int i = 0; i < height; i++) {
            final String line = i < source.size() ? source.get(i) : "";
            out.add(fit(line, width));
        }
        return out;
    }

    /**
     * Skip over an ANSI code sequence.
     *
     * @param text The text
     * @param start The starting position
     * @return The position after the ANSI code
     */
    private static int skipAnsi(final String text, final int start) {
        if (start + 1 < text.length() && text.charAt(start + 1) == '[') {
            for (int i = start + 2; i < text.length(); i++) {
                if (text.charAt(i) == 'm') {
                    return i + 1;
                }
            }
        }
        return start + 1;
    }
}
