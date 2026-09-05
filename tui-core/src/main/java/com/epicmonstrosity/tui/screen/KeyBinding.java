package com.epicmonstrosity.tui.screen;

/**
 * Describes a single key binding for a screen.
 *
 * @param key     The key string as received by handleInput() (e.g. "UP", "q", "ENTER")
 * @param display The display glyph/name shown in footer (e.g. "↑↓", "q")
 * @param label   Human-readable label shown in footer (e.g. "navigate", "quit")
 * @param primary Whether this binding is important enough to show in a space-constrained footer
 * @param action  The action to run when the binding is triggered
 */
public record KeyBinding(String key, String display, String label, boolean primary, Runnable action) {

    /**
     * Create a primary key binding.
     *
     * @param key The key
     * @param display The display glyph
     * @param label The label
     * @param action The action
     * @return The KeyBinding
     */
    public static KeyBinding primary(String key, String display, String label, Runnable action) {
        return new KeyBinding(key, display, label, true, action);
    }

    /**
     * Create a secondary key binding.
     *
     * @param key The key
     * @param display The display glyph
     * @param label The label
     * @param action The action
     * @return The KeyBinding
     */
    public static KeyBinding secondary(String key, String display, String label, Runnable action) {
        return new KeyBinding(key, display, label, false, action);
    }

    /**
     * Create a primary key binding with no action.
     *
     * @param key The key
     * @param display The display glyph
     * @param label The label
     * @return The KeyBinding
     */
    public static KeyBinding primary(String key, String display, String label) {
        return new KeyBinding(key, display, label, true, () -> {});
    }

    /**
     * Create a secondary key binding with no action.
     *
     * @param key The key
     * @param display The display glyph
     * @param label The label
     * @return The KeyBinding
     */
    public static KeyBinding secondary(String key, String display, String label) {
        return new KeyBinding(key, display, label, false, () -> {});
    }

    /**
     * Get the formatted string for display.
     *
     * @return The formatted string
     */
    public String formatted() {
        return "[" + display + "] " + label;
    }
}
