package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.screen.KeyBinding;

import java.util.List;

/**
 * A screen is a UI component that renders to the terminal and handles input.
 * Screens can be stacked, with new screens pushed on top of existing ones.
 */
public interface Screen {
    /**
     * Render the screen to an array of three strings: header, body, footer.
     *
     * @param ctx The view context with terminal dimensions and available body rows
     * @return An array with [header, body, footer] strings
     */
    String[] render(ViewContext ctx);

    /**
     * Handle a key press on the screen.
     *
     * @param key The key that was pressed
     * @return The screen to display after handling the key (can be the same screen)
     */
    Screen handleInput(String key);

    /**
     * Called when this screen is pushed onto the screen stack.
     *
     * @param host The UI host providing access to navigation, toasts, and events
     */
    void onEnter(UiHost host);

    /**
     * Called when this screen is popped from the screen stack.
     */
    void onExit();

    /**
     * Handle a UI event (paste, resize, tick, etc.).
     *
     * @param msg The UI message to handle
     */
    void onEvent(UiMessage msg);

    /**
     * Check if the screen should be invalidated and redrawn.
     *
     * @return true if the screen needs to be redrawn
     */
    boolean isInvalidated();

    /**
     * Get the key bindings for this screen.
     *
     * @return List of key bindings defined for this screen
     */
    default List<KeyBinding> keyBindings() {
        return List.of();
    }

    /**
     * When true, {@link TuiApp} skips the global key hook so the screen can
     * consume the key (for example typing {@code ?} in chat).
     *
     * @param key The key to check
     * @return true if the screen should capture the key
     */
    default boolean capturesGlobalKey(final String key) {
        return false;
    }
}
