package com.epicmonstrosity.tui.ansi;

/**
 * ANSI escape code constants for terminal styling.
 * Provides pre-defined style sequences for common TUI needs.
 */
public final class Ansi {
    /** Reset all styles to default. */
    public static final String RESET  = "\u001B[0m";

    /** Bold text. */
    public static final String BOLD   = "\u001B[1m";

    /** Dimmed text. */
    public static final String DIM    = "\u001B[2m";

    /** Green text. */
    public static final String GREEN  = "\u001B[32m";

    /** Yellow text. */
    public static final String YELLOW = "\u001B[33m";

    /** Red text. */
    public static final String RED    = "\u001B[31m";

    /** Cyan text. */
    public static final String CYAN   = "\u001B[36m";

    /** Blue text. */
    public static final String BLUE   = "\u001B[34m";

    /** Magenta text. */
    public static final String MAGENTA = "\u001B[35m";

    /** Reverse video (selection highlight). */
    public static final String BG_SEL = "\u001B[7m";

    /** Light gray background. */
    public static final String BG_LT_GREY = "\u001B[48;5;250m";

    /** Gray background. */
    public static final String BG_GREY = "\u001B[48;5;236m";

    /** Bright background. */
    public static final String BG_BRIGHT = "\u001B[48;5;196m";

    /** Dark gray background. */
    public static final String BG_DARK_GREY = "\u001B[48;5;235m";

    /** White text. */
    public static final String WHITE = "\u001B[97m";

    /** Light blue text. */
    public static final String L_BLUE = "\u001B[94m";

    private Ansi() { }
}
