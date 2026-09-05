package com.epicmonstrosity.tui;

/**
 * Interface for screen stack navigation and event delivery.
 * Implemented by {@link ScreenManager}.
 */
public interface ScreenNavigation {
    Screen peek();
    void push(Screen screen);
    void pop();
    void replace(Screen screen);

    /**
     * Deliver {@code msg} now on the calling thread. UI-thread only.
     * Workers must use {@link UiHost#post(UiMessage)}.
     *
     * @param msg The event to deliver
     */
    void forwardEvent(UiMessage msg);

    /**
     * Deliver {@code msg} to the screen beneath the current one. UI-thread only.
     *
     * @param msg The event to bubble to the screen below
     */
    void bubbleEvent(UiMessage msg);
}
