package com.epicmonstrosity.tui.screen;

import com.epicmonstrosity.tui.Screen;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.ansi.Ansi;
import com.epicmonstrosity.tui.ansi.DisplayWidth;
import com.epicmonstrosity.tui.ansi.Theme;

import java.util.List;

/**
 * Abstract template for building screens with header, body, and footer.
 */
public abstract class ScreenTemplate implements Screen {
    /** Reset ANSI code. */
    public static final String RESET = Ansi.RESET;
    /** Bold ANSI code. */
    public static final String BOLD = Ansi.BOLD;
    /** Dim ANSI code. */
    public static final String DIM = Ansi.DIM;
    /** Green ANSI code. */
    public static final String GREEN = Ansi.GREEN;
    /** Yellow ANSI code. */
    public static final String YELLOW = Ansi.YELLOW;
    /** Red ANSI code. */
    public static final String RED = Ansi.RED;
    /** Cyan ANSI code. */
    public static final String CYAN = Ansi.CYAN;
    /** Blue ANSI code. */
    public static final String BLUE = Ansi.BLUE;
    /** Magenta ANSI code. */
    public static final String MAGENTA = Ansi.MAGENTA;
    /** Selection background ANSI code. */
    public static final String BG_SEL = Ansi.BG_SEL;
    /** Light gray background ANSI code. */
    public static final String BG_LT_GREY = Ansi.BG_LT_GREY;
    /** Gray background ANSI code. */
    public static final String BG_GREY = Ansi.BG_GREY;
    /** Bright background ANSI code. */
    public static final String BG_BRIGHT = Ansi.BG_BRIGHT;
    /** Dark gray background ANSI code. */
    public static final String BG_DARK_GREY = Ansi.BG_DARK_GREY;
    /** White ANSI code. */
    public static final String WHITE = Ansi.WHITE;
    /** Light blue ANSI code. */
    public static final String L_BLUE = Ansi.L_BLUE;

    /**
     * Build the screen layout.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    protected abstract ScreenLayout buildLayout(ViewContext ctx);

    /**
     * Declare all key bindings this screen supports.
     * Primary bindings are shown first; secondary bindings fill the remaining space.
     */
    @Override
    public abstract List<KeyBinding> keyBindings();

    /**
     * Render the screen to an array of header, body, footer strings.
     *
     * @param ctx The view context
     * @return The rendered screen parts
     */
    @Override
    public final String[] render(final ViewContext ctx) {
        final ScreenLayout layout = buildLayout(ctx);
        String header = layout.header() == null ? "" : layout.header();
        if (layout.statusLine() != null && !layout.statusLine().isEmpty()) {
            header = header + layout.statusLine() + "\n";
        }
        final String footer = buildFooter(layout, ctx.terminalCols());
        return new String[]{header, layout.body(), footer};
    }

    /**
     * Handle a key by finding and executing the matching binding.
     *
     * @param key The key to handle
     * @return This screen
     */
    @Override
    public final Screen handleInput(final String key) {
        keyBindings().stream()
                .filter(b -> b.key().equals(key))
                .findFirst()
                .ifPresent(b -> b.action().run());
        return this;
    }

    /**
     * Build a simple header with title and optional badge.
     *
     * @param title The title
     * @param queueBadge The queue badge, or null
     * @return The header string
     */
    public String header(final String title, final String queueBadge) {
        final Theme theme = Theme.current();
        return theme.bold() + theme.title() + "\n  " + title + (queueBadge != null ? queueBadge : "") + theme.reset() + "\n"
                + theme.dim() + "  " + "=".repeat(DisplayWidth.of(title)) + theme.reset() + "\n\n";
    }

    /**
     * Build a bordered header with title and optional badge.
     *
     * @param title The title
     * @param queueBadge The queue badge, or null
     * @param cols The terminal columns
     * @return The bordered header
     */
    public String headerBordered(final String title, final String queueBadge, final int cols) {
        final Theme theme = Theme.current();
        final String content = " " + title + (queueBadge != null ? queueBadge : "");
        final String padding = " ".repeat(Math.max(0, cols - DisplayWidth.of(content)));
        final String fullPadding = " ".repeat(Math.max(0, cols));
        return    theme.title() + " │" + theme.reset() + theme.headerBg() + theme.headerFg() + theme.bold() + fullPadding + theme.reset() + "\n"
                + theme.title() + " │" + theme.reset() + theme.headerBg() + theme.headerFg() + theme.bold() + content + padding + theme.reset() + "\n"
                + theme.title() + " │" + theme.reset() + theme.headerBg() + theme.headerFg() + theme.bold() + fullPadding + theme.reset() + "\n\n";
    }

    /**
     * Append a detail row to a StringBuilder.
     *
     * @param sb The StringBuilder to append to
     * @param label The label
     * @param value The value
     */
    public void detailRow(final StringBuilder sb, final String label, final String value) {
        sb.append(String.format("  %s%-16s%s %s\n", BOLD, label + ":", RESET, value));
    }

    /**
     * Configure a field row in a config form.
     *
     * @param sb The StringBuilder to append to
     * @param cursor The cursor position
     * @param idx The field index
     * @param label The field label
     * @param value The field value
     */
    public void configField(final StringBuilder sb, final int cursor, final int idx, final String label, final String value) {
        final boolean sel = (cursor == idx);
        if (sel) sb.append(Theme.current().selection()).append(Theme.current().bold());
        sb.append(String.format("  %-30s %s", label + ":", value));
        if (sel) sb.append(RESET);
        sb.append("\n");
    }

    /**
     * Build a scroll indicator string.
     *
     * @param offset The current offset
     * @param viewportH The viewport height
     * @param total The total items
     * @return The scroll indicator
     */
    public String scrollIndicator(final int offset, final int viewportH, final int total) {
        return String.format("  ↕ %d–%d of %d", offset + 1,
                Math.min(total, offset + viewportH), total);
    }

    /**
     * Build an error footer string.
     *
     * @param message The error message, or empty
     * @return The error footer
     */
    public String footerError(final String message) {
        if (message == null || message.isEmpty()) return "";
        final Theme theme = Theme.current();
        return "  " + theme.error() + message + theme.reset();
    }

    /**
     * Truncate a string to a maximum width.
     *
     * @param s The string to truncate
     * @param max The maximum width
     * @return The truncated string
     */
    public String truncate(final String s, final int max) {
        if (s == null)
            return rpad("", max, ' ');
        if (DisplayWidth.of(s) <= max) {
            return rpad(s, max, ' ');
        }
        return rpad(DisplayWidth.prefix(s, Math.max(0, max - 1)) + "…", max, ' ');
    }

    /**
     * Right-pad a string to a length.
     *
     * @param str The string to pad
     * @param length The target length
     * @param padChar The padding character
     * @return The padded string
     */
    public String rpad(final String str, final int length, final char padChar) {
        final String normalized = str == null ? "" : str.trim();
        final int width = DisplayWidth.of(normalized);
        if (width >= length) {
            return DisplayWidth.prefix(normalized, length);
        }
        return normalized + String.valueOf(padChar).repeat(length - width);
    }

    /**
     * Build the footer from key bindings and error messages.
     *
     * @param layout The screen layout
     * @param cols The terminal columns
     * @return The footer string
     */
    private String buildFooter(final ScreenLayout layout, final int cols) {
        final int errorLen = layout.footerError().isEmpty() ? 0
                : DisplayWidth.of(stripAnsi(layout.footerError())) + 2;
        final int available = cols - 4 - errorLen;

        final StringBuilder sb = new StringBuilder(DIM + "  ");

        final List<KeyBinding> primaries   = keyBindings().stream().filter(KeyBinding::primary).toList();
        final List<KeyBinding> secondaries = keyBindings().stream().filter(b -> !b.primary()).toList();

        final int used = appendBindings(sb, primaries, available);
        appendBindings(sb, secondaries, available - used);

        if (!layout.footerError().isEmpty()) {
            sb.append("  ").append(layout.footerError());
        }

        sb.append(RESET);
        return sb.toString();
    }

    /**
     * Append key bindings to a StringBuilder.
     *
     * @param sb The StringBuilder to append to
     * @param bindings The bindings to append
     * @param budget The available budget
     * @return The used budget
     */
    private int appendBindings(final StringBuilder sb, final List<KeyBinding> bindings, final int budget) {
        int used = 0;
        for (final KeyBinding b : bindings) {
            final String fragment = b.formatted() + "  ";
            final int fragmentWidth = DisplayWidth.of(stripAnsi(fragment));
            if (used + fragmentWidth > budget) break;
            sb.append(fragment);
            used += fragmentWidth;
        }
        return used;
    }

    /**
     * Strip ANSI codes from a string.
     *
     * @param s The string to strip
     * @return The stripped string
     */
    private static String stripAnsi(final String s) {
        return s.replaceAll("\u001B\\[[\\d;]*m", "");
    }
}
