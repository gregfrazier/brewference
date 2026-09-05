package com.epicmonstrosity.tui.ansi;

import java.util.ArrayList;
import java.util.List;

/**
 * Terminal column widths. CJK / fullwidth / emoji occupy two columns;
 * combining marks, format characters, and C0/C1 controls occupy none.
 */
public final class DisplayWidth {
    private DisplayWidth() {
    }

    /**
     * Get the display width of a single code point.
     * CJK / fullwidth / emoji occupy two columns; combining marks, format
     * characters, and C0/C1 controls occupy none.
     *
     * @param codePoint The Unicode code point
     * @return 0 for zero-width chars, 1 for normal, 2 for wide chars
     */
    public static int of(final int codePoint) {
        if (codePoint == 0 || codePoint < 32 || (codePoint >= 0x7f && codePoint < 0xa0)) {
            return 0;
        }
        if (isZeroWidth(codePoint)) {
            return 0;
        }
        return isWide(codePoint) ? 2 : 1;
    }

    /**
     * Get the display width of a text string.
     *
     * @param text The text to measure
     * @return The total width in columns, or 0 for null/empty
     */
    public static int of(final CharSequence text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        int width = 0;
        for (int i = 0; i < text.length(); ) {
            final int cp = Character.codePointAt(text, i);
            width += of(cp);
            i += Character.charCount(cp);
        }
        return width;
    }

    /**
     * Return the prefix of text that fits within maxColumns.
     *
     * @param text The text to truncate
     * @param maxColumns The maximum column width
     * @return The truncated text, or empty string if text is null/empty or maxColumns &lt;= 0
     */
    public static String prefix(final CharSequence text, final int maxColumns) {
        if (text == null || maxColumns <= 0) {
            return "";
        }
        final StringBuilder out = new StringBuilder();
        int width = 0;
        for (int i = 0; i < text.length(); ) {
            final int cp = Character.codePointAt(text, i);
            final int n = Character.charCount(cp);
            final int w = of(cp);
            if (width + w > maxColumns) {
                break;
            }
            out.append(text, i, i + n);
            width += w;
            i += n;
        }
        return out.toString();
    }

    /**
     * Wrap text to fit within the given column width.
     *
     * @param text The text to wrap
     * @param columns The maximum column width per line
     * @return List of wrapped lines, or single empty string if null/empty
     */
    public static List<String> wrap(final String text, final int columns) {
        if (text == null || text.isEmpty()) {
            return List.of("");
        }
        final int cols = Math.max(1, columns);
        final List<String> lines = new ArrayList<>();
        final StringBuilder line = new StringBuilder();
        int width = 0;
        for (int i = 0; i < text.length(); ) {
            final int cp = text.codePointAt(i);
            final int n = Character.charCount(cp);
            final int w = of(cp);
            if (w > cols) {
                if (!line.isEmpty()) {
                    lines.add(line.toString());
                    line.setLength(0);
                    width = 0;
                }
                lines.add(text.substring(i, i + n));
                i += n;
                continue;
            }
            if (width + w > cols && !line.isEmpty()) {
                lines.add(line.toString());
                line.setLength(0);
                width = 0;
            }
            line.appendCodePoint(cp);
            width += w;
            i += n;
        }
        if (!line.isEmpty()) {
            lines.add(line.toString());
        }
        return lines;
    }

    /**
     * Check if a code point is zero-width (does not advance cursor).
     *
     * @param cp The code point to check
     * @return true if the character is zero-width
     */
    private static boolean isZeroWidth(final int cp) {
        if (cp == 0x00AD) {
            return false;
        }
        final int type = Character.getType(cp);
        if (type == Character.NON_SPACING_MARK
                || type == Character.COMBINING_SPACING_MARK
                || type == Character.ENCLOSING_MARK
                || type == Character.FORMAT) {
            return true;
        }
        return (cp >= 0x1160 && cp <= 0x11FF)
                || (cp >= 0x200B && cp <= 0x200F)
                || (cp >= 0x202A && cp <= 0x202E)
                || (cp >= 0x2060 && cp <= 0x206F)
                || cp == 0xFEFF
                || (cp >= 0xE0100 && cp <= 0xE01EF);
    }

    /**
     * Check if a code point is wide (occupies 2 columns).
     *
     * @param cp The code point to check
     * @return true if the character is wide
     */
    private static boolean isWide(final int cp) {
        if (cp >= 0x1100 && cp <= 0x115F) {
            return true;
        }
        if (cp == 0x2329 || cp == 0x232A) {
            return true;
        }
        if (cp >= 0x2E80 && cp <= 0xA4CF) {
            return true;
        }
        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            return true;
        }
        if (cp >= 0xF900 && cp <= 0xFAFF) {
            return true;
        }
        if (cp >= 0xFE10 && cp <= 0xFE19) {
            return true;
        }
        if (cp >= 0xFE30 && cp <= 0xFE6F) {
            return true;
        }
        if (cp >= 0xFF00 && cp <= 0xFF60) {
            return true;
        }
        if (cp >= 0xFFE0 && cp <= 0xFFE6) {
            return true;
        }
        if (cp >= 0x1F1E6 && cp <= 0x1F1FF) {
            return true;
        }
        if (cp >= 0x1F300 && cp <= 0x1FAFF) {
            return true;
        }
        return isWideSymbol(cp);
    }

    /**
     * Check if a symbol code point is wide (occupies 2 columns).
     *
     * @param cp The symbol code point to check
     * @return true if the symbol is wide
     */
    private static boolean isWideSymbol(final int cp) {
        return (cp >= 0x231A && cp <= 0x231B)
                || (cp >= 0x23E9 && cp <= 0x23EC)
                || cp == 0x23F0 || cp == 0x23F3
                || (cp >= 0x25FD && cp <= 0x25FE)
                || (cp >= 0x2614 && cp <= 0x2615)
                || (cp >= 0x2648 && cp <= 0x2653)
                || cp == 0x267F || cp == 0x2693 || cp == 0x26A1
                || (cp >= 0x26AA && cp <= 0x26AB)
                || (cp >= 0x26BD && cp <= 0x26BE)
                || (cp >= 0x26C4 && cp <= 0x26C5)
                || cp == 0x26CE || cp == 0x26D4 || cp == 0x26EA
                || (cp >= 0x26F2 && cp <= 0x26F3)
                || cp == 0x26F5 || cp == 0x26FA || cp == 0x26FD
                || cp == 0x2705 || (cp >= 0x270A && cp <= 0x270B)
                || cp == 0x2728 || cp == 0x274C || cp == 0x274E
                || (cp >= 0x2753 && cp <= 0x2755) || cp == 0x2757
                || (cp >= 0x2795 && cp <= 0x2797)
                || cp == 0x27B0 || cp == 0x27BF
                || (cp >= 0x2B1B && cp <= 0x2B1C)
                || cp == 0x2B50 || cp == 0x2B55;
    }
}
