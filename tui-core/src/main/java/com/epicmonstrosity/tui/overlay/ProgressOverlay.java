package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Overlay;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;

import java.util.ArrayList;
import java.util.List;

/**
 * A progress overlay showing a progress bar or spinner.
 */
public final class ProgressOverlay implements Overlay {
    private static final String SPINNER = "|/-\\";

    private final String title;
    private final CloseMe closeMe = new CloseMe();
    private String message;
    private int current;
    private int total;
    private boolean indeterminate;
    private Runnable onCancel;
    private UiHost host;
    private boolean invalidated = true;
    private boolean closed;

    /**
     * Construct a progress overlay.
     *
     * @param title The overlay title
     * @param message The progress message
     * @param current The current progress
     * @param total The total progress
     * @param indeterminate Whether the progress is indeterminate
     */
    private ProgressOverlay(final String title,
                            final String message,
                            final int current,
                            final int total,
                            final boolean indeterminate) {
        this.title = title == null || title.isBlank() ? "Progress" : title;
        this.message = message == null ? "" : message;
        this.current = Math.max(0, current);
        this.total = Math.max(0, total);
        this.indeterminate = indeterminate;
    }

    /**
     * Create a determinate progress overlay.
     *
     * @param title The overlay title
     * @param message The progress message
     * @param current The current progress
     * @param total The total progress
     * @return The ProgressOverlay
     */
    public static ProgressOverlay determinate(final String title,
                                              final String message,
                                              final int current,
                                              final int total) {
        return new ProgressOverlay(title, message, current, total, false);
    }

    /**
     * Create an indeterminate progress overlay.
     *
     * @param title The overlay title
     * @param message The progress message
     * @return The ProgressOverlay
     */
    public static ProgressOverlay indeterminate(final String title, final String message) {
        return new ProgressOverlay(title, message, 0, 0, true);
    }

    /**
     * Set an on-cancel callback.
     *
     * @param onCancel The callback to run on cancel
     * @return This ProgressOverlay for chaining
     */
    public ProgressOverlay onCancel(final Runnable onCancel) {
        this.onCancel = onCancel;
        return this;
    }

    /**
     * Set the progress message.
     *
     * @param message The message
     */
    public synchronized void setMessage(final String message) {
        this.message = message == null ? "" : message;
        this.invalidated = true;
    }

    /**
     * Set the progress values.
     *
     * @param current The current progress
     * @param total The total progress
     */
    public synchronized void setProgress(final int current, final int total) {
        this.current = Math.max(0, current);
        this.total = Math.max(0, total);
        this.indeterminate = false;
        this.invalidated = true;
    }

    /**
     * Set whether the progress is indeterminate.
     *
     * @param indeterminate Whether indeterminate
     */
    public synchronized void setIndeterminate(final boolean indeterminate) {
        this.indeterminate = indeterminate;
        this.invalidated = true;
    }

    /**
     * Mark the progress as complete.
     */
    public void complete() {
        if (host != null) {
            host.post(closeMe);
        }
    }

    /**
     * Called when the progress overlay opens.
     *
     * @param host The UI host
     */
    @Override
    public void onOpen(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Render the progress overlay body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @return The rendered overlay body
     */
    @Override
    public synchronized String renderBody(final ViewContext ctx, final String underlyingBody) {
        invalidated = false;
        final List<String> rows = new ArrayList<>();
        if (!message.isBlank()) {
            rows.add(message);
        }
        rows.add(indeterminate ? spinnerLine() : barLine());
        if (onCancel != null) {
            rows.add("");
            rows.add("[Esc] cancel");
        }
        return ModalBox.paint(ctx, underlyingBody, title, rows);
    }

    /**
     * Handle events for the progress overlay.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (closed) {
            return;
        }
        if (msg instanceof CloseMe) {
            close();
            return;
        }
        if (msg instanceof KeyPressed(String key) && "ESC".equals(key) && onCancel != null) {
            close();
            onCancel.run();
        }
    }

    /**
     * Check if the progress overlay should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public synchronized boolean isInvalidated() {
        return invalidated || (indeterminate && !closed);
    }

    /**
     * Close the progress overlay.
     */
    private void close() {
        closed = true;
        if (host != null) {
            host.popOverlay();
        }
    }

    /**
     * Get the progress bar line.
     *
     * @return The bar line
     */
    private String barLine() {
        final int width = 20;
        final int safeTotal = Math.max(1, total);
        final int filled = Math.min(width, (int) Math.round(width * (current / (double) safeTotal)));
        return "[" + "#".repeat(filled) + "-".repeat(width - filled) + "] " + current + "/" + total;
    }

    /**
     * Get the spinner line.
     *
     * @return The spinner line
     */
    private String spinnerLine() {
        final int frame = (int) ((System.nanoTime() / 120_000_000L) % SPINNER.length());
        return "[" + SPINNER.charAt(frame) + "] working…";
    }

    /**
     * A close message posted to the host.
     */
    private final class CloseMe implements UiMessage {
    }
}
