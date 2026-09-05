package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.overlay.Toast;

/**
 * Interface for delivering UI events and managing toasts/overlays.
 * Implemented by {@link ScreenManager}.
 */
public interface UiHost {
    /**
     * Get the screen navigation instance.
     *
     * @return The ScreenNavigation instance
     */
    ScreenNavigation navigation();

    /**
     * Get the terminal rectangle.
     *
     * @return The TerminalRect instance
     */
    TerminalRect terminal();

    /**
     * Set the error banner text displayed in the footer.
     *
     * @param message The error message, or null to clear
     */
    void setErrorBanner(String message);

    /**
     * Get the current error banner text.
     *
     * @return The error banner message
     */
    String errorBanner();

    /**
     * Push an overlay onto the modal overlay stack.
     *
     * @param overlay The overlay to push
     */
    void pushOverlay(Overlay overlay);

    /**
     * Remove the top overlay from the stack.
     */
    void popOverlay();

    /**
     * Get the current (top) overlay without removing it.
     *
     * @return The current overlay, or null if stack is empty
     */
    Overlay peekOverlay();

    /**
     * Queue a timed status overlay. Does not steal keys and is not on the
     * modal overlay stack. Several toasts stack above the footer; overflow
     * waits until a slot frees.
     *
     * @param toast The toast to show
     */
    void showToast(Toast toast);

    /**
     * Show an info toast with the given message.
     *
     * @param message The message to display
     */
    default void showToast(final String message) {
        showToast(Toast.info(message));
    }

    /**
     * Get the newest visible toast, or the sticky status if no toasts are visible.
     *
     * @return The visible toast or status, or null if none
     */
    Toast peekToast();

    /**
     * Dismiss the newest visible toast.
     */
    void dismissToast();

    /**
     * Get the number of toasts that are queued but not yet visible.
     *
     * @return The count of pending toasts
     */
    int pendingToastCount();

    /**
     * Sticky status overlay. Stays until {@link #clearStatus()} and stacks
     * under live toasts. Use this for scan/copy progress instead of {@link #setErrorBanner}.
     *
     * @param status The status toast to set, or null to clear
     */
    void setStatus(Toast status);

    /**
     * Set a sticky status toast with the given message.
     *
     * @param message The message to display
     */
    default void setStatus(final String message) {
        setStatus(new Toast(message, Toast.Kind.INFO, 0));
    }

    /**
     * Clear the sticky status toast.
     */
    void clearStatus();

    /**
     * Get the current sticky status toast.
     *
     * @return The status toast, or null if none
     */
    Toast peekStatus();

    /**
     * Live toasts currently painted, oldest first. Does not include status.
     *
     * @return List of visible toasts
     */
    java.util.List<Toast> visibleToasts();

    /**
     * Get the newest live toast, or the sticky status if no toasts visible.
     *
     * @return The visible toast or status, or null if none
     */
    Toast visibleToast();

    /**
     * Queue {@code message} from any thread. {@link TuiApp} delivers it on the
     * UI thread via {@link ScreenNavigation#forwardEvent(UiMessage)} and
     * redraws. Use this instead of calling {@code forwardEvent} from a worker.
     *
     * @param message The message to queue
     */
    void post(UiMessage message);
}
