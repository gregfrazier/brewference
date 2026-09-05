package com.epicmonstrosity.tui;

/**
 * Modal painted over the current screen's body. Does not participate in the
 * screen stack and must not trigger {@link Screen#onExit()}.
 */
public interface Overlay {
    /**
     * Called when the overlay is pushed onto the stack.
     *
     * @param host The UI host containing the screen and event queue
     */
    default void onOpen(final UiHost host) { }

    /**
     * Render the overlay body over the underlying screen body.
     *
     * @param ctx The view context with terminal dimensions
     * @param underlyingBody The screen's body content
     * @return The overlay's body content to paint over the screen
     */
    String renderBody(ViewContext ctx, String underlyingBody);

    /**
     * Handle a UI event. The overlay receives events before the screen.
     *
     * @param msg The UI message to handle
     */
    void onEvent(UiMessage msg);

    /**
     * Check if the overlay should be invalidated and redrawn.
     *
     * @return true if the overlay needs to be redrawn
     */
    boolean isInvalidated();
}
