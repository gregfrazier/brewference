package com.epicmonstrosity.tui.ansi;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Named color roles for kit chrome. Swap with {@link #setCurrent(Theme)}
 * or {@code TuiApp.theme(...)} so apps are not stuck with the default greens.
 */
public final class Theme {
    public static final Theme DEFAULT = new Theme(
            Ansi.RESET,
            Ansi.BOLD,
            Ansi.DIM,
            Ansi.GREEN,
            Ansi.L_BLUE,
            Ansi.DIM,
            Ansi.GREEN,
            Ansi.YELLOW,
            Ansi.RED,
            Ansi.CYAN,
            Ansi.BG_SEL,
            Ansi.WHITE,
            Ansi.BG_GREY
    );

    public static final Theme MONO = new Theme(
            Ansi.RESET,
            Ansi.BOLD,
            Ansi.DIM,
            "",
            "",
            Ansi.DIM,
            "",
            "",
            "",
            "",
            Ansi.BG_SEL,
            "",
            ""
    );

    public static final Theme AMBER = new Theme(
            Ansi.RESET,
            Ansi.BOLD,
            Ansi.DIM,
            Ansi.YELLOW,
            Ansi.YELLOW,
            Ansi.DIM,
            Ansi.YELLOW,
            Ansi.YELLOW,
            Ansi.RED,
            Ansi.YELLOW,
            Ansi.BG_SEL,
            Ansi.YELLOW,
            Ansi.BG_DARK_GREY
    );

    private static final AtomicReference<Theme> CURRENT = new AtomicReference<>(DEFAULT);

    /** Reset color code. */
    private final String reset;
    /** Bold color code. */
    private final String bold;
    /** Dimmed color code. */
    private final String dim;
    /** Accent color code. */
    private final String accent;
    /** Title color code. */
    private final String title;
    /** Muted color code. */
    private final String muted;
    /** Success color code. */
    private final String success;
    /** Warning color code. */
    private final String warning;
    /** Error color code. */
    private final String error;
    /** Info color code. */
    private final String info;
    /** Selection color code. */
    private final String selection;
    /** Header foreground color code. */
    private final String headerFg;
    /** Header background color code. */
    private final String headerBg;

    /**
     * Construct a theme with the given ANSI color codes.
     *
     * @param reset Reset code
     * @param bold Bold code
     * @param dim Dim code
     * @param accent Accent code
     * @param title Title code
     * @param muted Muted code
     * @param success Success code
     * @param warning Warning code
     * @param error Error code
     * @param info Info code
     * @param selection Selection code
     * @param headerFg Header foreground code
     * @param headerBg Header background code
     */
    public Theme(final String reset,
                 final String bold,
                 final String dim,
                 final String accent,
                 final String title,
                 final String muted,
                 final String success,
                 final String warning,
                 final String error,
                 final String info,
                 final String selection,
                 final String headerFg,
                 final String headerBg) {
        this.reset = nz(reset);
        this.bold = nz(bold);
        this.dim = nz(dim);
        this.accent = nz(accent);
        this.title = nz(title);
        this.muted = nz(muted);
        this.success = nz(success);
        this.warning = nz(warning);
        this.error = nz(error);
        this.info = nz(info);
        this.selection = nz(selection);
        this.headerFg = nz(headerFg);
        this.headerBg = nz(headerBg);
    }

    /**
     * Get the current active theme.
     *
     * @return The current theme
     */
    public static Theme current() {
        return CURRENT.get();
    }

    /**
     * Set the current active theme.
     *
     * @param theme The new theme, or null to use DEFAULT
     * @return The set theme
     */
    public static Theme setCurrent(final Theme theme) {
        final Theme next = theme == null ? DEFAULT : theme;
        CURRENT.set(next);
        return next;
    }

    /**
     * Paint text with the given style.
     *
     * @param style The style to apply
     * @param text The text to paint, or null for empty
     * @return The styled text
     */
    public String paint(final String style, final String text) {
        return nz(style) + (text == null ? "" : text) + reset;
    }

    /**
     * Get the reset code.
     *
     * @return The reset ANSI code
     */
    public String reset() {
        return reset;
    }

    /**
     * Get the bold code.
     *
     * @return The bold ANSI code
     */
    public String bold() {
        return bold;
    }

    /**
     * Get the dim code.
     *
     * @return The dim ANSI code
     */
    public String dim() {
        return dim;
    }

    /**
     * Get the accent code.
     *
     * @return The accent ANSI code
     */
    public String accent() {
        return accent;
    }

    /**
     * Get the title code.
     *
     * @return The title ANSI code
     */
    public String title() {
        return title;
    }

    /**
     * Get the muted code.
     *
     * @return The muted ANSI code
     */
    public String muted() {
        return muted;
    }

    /**
     * Get the success code.
     *
     * @return The success ANSI code
     */
    public String success() {
        return success;
    }

    /**
     * Get the warning code.
     *
     * @return The warning ANSI code
     */
    public String warning() {
        return warning;
    }

    /**
     * Get the error code.
     *
     * @return The error ANSI code
     */
    public String error() {
        return error;
    }

    /**
     * Get the info code.
     *
     * @return The info ANSI code
     */
    public String info() {
        return info;
    }

    /**
     * Get the selection code.
     *
     * @return The selection ANSI code
     */
    public String selection() {
        return selection;
    }

    /**
     * Get the header foreground code.
     *
     * @return The header foreground ANSI code
     */
    public String headerFg() {
        return headerFg;
    }

    /**
     * Get the header background code.
     *
     * @return The header background ANSI code
     */
    public String headerBg() {
        return headerBg;
    }

    /**
     * Return the value or empty string if null.
     *
     * @param value The value to return
     * @return The value or empty string
     */
    private static String nz(final String value) {
        return value == null ? "" : value;
    }
}
