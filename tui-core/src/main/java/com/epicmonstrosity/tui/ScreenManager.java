package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.overlay.Toast;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Manages the screen stack, overlay stack, toast notifications, and event queue.
 * Implements both {@link ScreenNavigation} for screen management and {@link UiHost}
 * for event delivery and toast handling.
 */
public class ScreenManager implements ScreenNavigation, UiHost {
    static final int MAX_VISIBLE_TOASTS = 3;

    private final Deque<Screen> screenStack = new ArrayDeque<>();
    private final Deque<Overlay> overlayStack = new ArrayDeque<>();
    private final ConcurrentLinkedQueue<UiMessage> inbox = new ConcurrentLinkedQueue<>();
    private final Deque<Toast> toasts = new ArrayDeque<>();
    private final TerminalRect terminal;
    private String errorBanner = "";
    private Toast status;
    private boolean toastChromeChanged;

    public ScreenManager(final TerminalRect terminal) {
        this.terminal = terminal;
    }

    /**
     * Get the current (top) screen on the stack without removing it.
     *
     * @return The current screen, or null if stack is empty
     */
    @Override
    public Screen peek() {
        return screenStack.peek();
    }

    public boolean isCurrentScreenInvalidated() {
        if (!inbox.isEmpty()) {
            return true;
        }
        if (hasExpiredToast()) {
            return true;
        }
        final Overlay overlay = peekOverlay();
        if (overlay != null && overlay.isInvalidated()) {
            return true;
        }
        final Screen current = peek();
        return current != null && current.isInvalidated();
    }

    /**
     * Push a new screen onto the top of the stack.
     *
     * @param screen The screen to push
     */
    @Override
    public void push(final Screen screen) {
        overlayStack.clear();
        screenStack.push(screen);
        screen.onEnter(this);
    }

    /**
     * Remove the top screen from the stack.
     * The screen's {@code onExit()} method is called before removal.
     */
    @Override
    public void pop() {
        if (screenStack.size() <= 1) {
            return;
        }

        overlayStack.clear();
        final Screen removed = screenStack.pop();
        removed.onExit();

        final Screen current = screenStack.peek();
        if (current != null) {
            current.onEnter(this);
        }
    }

    /**
     * Replace the current screen with a new one.
     * The current screen's {@code onExit()} method is called.
     *
     * @param screen The new screen to display
     */
    @Override
    public void replace(final Screen screen) {
        overlayStack.clear();
        if (!screenStack.isEmpty()) {
            final Screen removed = screenStack.pop();
            removed.onExit();
        }
        screenStack.push(screen);
        screen.onEnter(this);
    }

    /**
     * Deliver an event to the current overlay, or the current screen if no overlay.
     * This is the primary event delivery path for UI-thread events.
     *
     * @param msg The UI event to deliver
     */
    @Override
    public void forwardEvent(final UiMessage msg) {
        final Overlay overlay = peekOverlay();
        if (overlay != null) {
            overlay.onEvent(msg);
            return;
        }
        final Screen current = peek();
        if (current != null) {
            current.onEvent(msg);
        }
    }

    /**
     * Deliver an event to the screen beneath the current one.
     * UI-thread only. Use {@link #forwardEvent(UiMessage)} for normal events.
     *
     * @param msg The UI event to bubble to the screen below
     */
    @Override
    public void bubbleEvent(final UiMessage msg) {
        if (screenStack.size() < 2) {
            return;
        }
        final Iterator<Screen> it = screenStack.iterator();
        it.next();
        it.next().onEvent(msg);
    }

    /**
     * Get the screen navigation instance (this manager).
     *
     * @return This ScreenManager instance
     */
    @Override
    public ScreenNavigation navigation() {
        return this;
    }

    /**
     * Get the terminal rectangle.
     *
     * @return The terminal dimensions
     */
    @Override
    public TerminalRect terminal() {
        return terminal;
    }

    /**
     * Set the error banner text displayed in the footer.
     *
     * @param message The error message, or null to clear
     */
    @Override
    public void setErrorBanner(final String message) {
        this.errorBanner = message == null ? "" : message;
    }

    /**
     * Get the current error banner text.
     *
     * @return The error banner message
     */
    @Override
    public String errorBanner() {
        return errorBanner;
    }

    /**
     * Push an overlay onto the modal overlay stack.
     *
     * @param overlay The overlay to push
     */
    @Override
    public void pushOverlay(final Overlay overlay) {
        overlayStack.push(overlay);
        overlay.onOpen(this);
    }

    /**
     * Remove the top overlay from the stack.
     */
    @Override
    public void popOverlay() {
        if (!overlayStack.isEmpty()) {
            overlayStack.pop();
        }
    }

    /**
     * Get the current (top) overlay without removing it.
     *
     * @return The current overlay, or null if stack is empty
     */
    @Override
    public Overlay peekOverlay() {
        return overlayStack.peek();
    }

    /**
     * Show a toast notification. Multiple toasts stack above the footer.
     * Use {@link #dismissToast()} to remove it.
     *
     * @param toast The toast to show
     */
    @Override
    public synchronized void showToast(final Toast toast) {
        if (toast == null) {
            return;
        }
        toasts.addLast(toast);
        revealToasts();
        toastChromeChanged = true;
    }

    /**
     * Get the newest visible toast, or the sticky status if no toasts are visible.
     *
     * @return The visible toast or status, or null if none
     */
    @Override
    public synchronized Toast peekToast() {
        final List<Toast> visible = visibleToasts();
        return visible.isEmpty() ? null : visible.getLast();
    }

    /**
     * Dismiss the newest visible toast.
     */
    @Override
    public synchronized void dismissToast() {
        final List<Toast> visible = visibleToasts();
        if (!visible.isEmpty()) {
            toasts.remove(visible.getLast());
        }
        revealToasts();
        toastChromeChanged = true;
    }

    /**
     * Get the number of toasts that are queued but not yet visible.
     *
     * @return The count of pending toasts
     */
    @Override
    public synchronized int pendingToastCount() {
        return Math.max(0, liveToastCount() - visibleToasts().size());
    }

    /**
     * Set a sticky status toast that stays until {@link #clearStatus()}.
     * Stacks under live toasts.
     *
     * @param status The status toast, or null to clear
     */
    @Override
    public synchronized void setStatus(final Toast status) {
        this.status = status == null ? null : new Toast(status.message(), status.kind(), 0);
        toastChromeChanged = true;
    }

    /**
     * Clear the sticky status toast.
     */
    @Override
    public synchronized void clearStatus() {
        status = null;
        toastChromeChanged = true;
    }

    /**
     * Get the current sticky status toast.
     *
     * @return The status toast, or null if none
     */
    @Override
    public synchronized Toast peekStatus() {
        return status;
    }

    /**
     * Get the list of currently visible toasts (not expired), oldest first.
     * Does not include the sticky status.
     *
     * @return List of visible toasts, or empty list if none
     */
    @Override
    public synchronized List<Toast> visibleToasts() {
        final List<Toast> visible = new ArrayList<>();
        for (final Toast toast : toasts) {
            if (toast.expired()) {
                continue;
            }
            visible.add(toast);
            if (visible.size() >= MAX_VISIBLE_TOASTS) {
                break;
            }
        }
        return List.copyOf(visible);
    }

    /**
     * Get the newest visible toast, or the sticky status if no toasts visible.
     *
     * @return The visible toast or status, or null if none
     */
    @Override
    public synchronized Toast visibleToast() {
        final List<Toast> visible = visibleToasts();
        if (!visible.isEmpty()) {
            return visible.getLast();
        }
        return status;
    }

    /**
     * UI-thread only. Remove expired toasts and reposition visible ones.
     *
     * @return true if any toast visibility changed
     */
    public synchronized boolean sweepToast() {
        boolean changed = toastChromeChanged;
        toastChromeChanged = false;
        final int before = toasts.size();
        toasts.removeIf(Toast::expired);
        if (toasts.size() != before) {
            changed = true;
        }
        if (revealToasts()) {
            changed = true;
        }
        return changed;
    }

    /**
     * Check if any toast in the queue has expired.
     *
     * @return true if an expired toast exists
     */
    private synchronized boolean hasExpiredToast() {
        if (toastChromeChanged) {
            return true;
        }
        for (final Toast toast : toasts) {
            if (toast.expired()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Count the number of non-expired toasts in the queue.
     *
     * @return The count of live toasts
     */
    private int liveToastCount() {
        int count = 0;
        for (final Toast toast : toasts) {
            if (!toast.expired()) {
                count++;
            }
        }
        return count;
    }

    /**
     * Mark visible toasts as shown and reveal them in the footer.
     *
     * @return true if any toast was newly shown
     */
    private boolean revealToasts() {
        boolean revealed = false;
        int shown = 0;
        for (final Toast toast : toasts) {
            if (toast.expired()) {
                continue;
            }
            if (shown >= MAX_VISIBLE_TOASTS) {
                break;
            }
            if (!toast.isShown()) {
                toast.markShown();
                revealed = true;
            }
            shown++;
        }
        return revealed;
    }

    /**
     * Queue a message to be delivered on the UI thread.
     * Thread-safe - use this from worker threads instead of {@link #forwardEvent(UiMessage)}.
     *
     * @param message The message to queue
     */
    @Override
    public void post(final UiMessage message) {
        if (message != null) {
            inbox.add(message);
        }
    }

    /**
     * UI-thread only. Deliver all queued messages and return the count.
     *
     * @return The number of messages delivered
     */
    public int drainPostedEvents() {
        int count = 0;
        UiMessage message;
        while ((message = inbox.poll()) != null) {
            forwardEvent(message);
            count++;
        }
        return count;
    }
}
