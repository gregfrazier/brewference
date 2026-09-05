package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.ansi.Theme;

/**
 * Timed status banner. Does not sit on the overlay stack and does not steal keys.
 * Duration {@code <= 0} stays until {@code UiHost.dismissToast()}.
 */
public final class Toast {
    /** Toast kinds. */
    public enum Kind { INFO, SUCCESS, WARNING, ERROR }

    /** Default toast duration in milliseconds. */
    public static final long DEFAULT_DURATION_MS = 2500L;

    private final String message;
    private final Kind kind;
    private final long durationMillis;
    private volatile long shownNanos;

    /**
     * Construct a toast.
     *
     * @param message The toast message
     * @param kind The toast kind
     * @param durationMillis The duration in milliseconds, or &lt;= 0 for indefinite
     */
    public Toast(final String message, final Kind kind, final long durationMillis) {
        this.message = message == null ? "" : message;
        this.kind = kind == null ? Kind.INFO : kind;
        this.durationMillis = durationMillis;
    }

    /**
     * Start the expiry clock the first time this toast is actually shown.
     * Queued / overflow toasts stay pending until then.
     */
    public void markShown() {
        if (shownNanos == 0L) {
            shownNanos = System.nanoTime();
        }
    }

    /**
     * Check if the toast has been shown.
     *
     * @return true if shown
     */
    public boolean isShown() {
        return shownNanos != 0L;
    }

    /**
     * Create an info toast.
     *
     * @param message The message
     * @return The Toast
     */
    public static Toast info(final String message) {
        return new Toast(message, Kind.INFO, DEFAULT_DURATION_MS);
    }

    /**
     * Create a success toast.
     *
     * @param message The message
     * @return The Toast
     */
    public static Toast success(final String message) {
        return new Toast(message, Kind.SUCCESS, DEFAULT_DURATION_MS);
    }

    /**
     * Create a warning toast.
     *
     * @param message The message
     * @return The Toast
     */
    public static Toast warning(final String message) {
        return new Toast(message, Kind.WARNING, DEFAULT_DURATION_MS);
    }

    /**
     * Create an error toast.
     *
     * @param message The message
     * @return The Toast
     */
    public static Toast error(final String message) {
        return new Toast(message, Kind.ERROR, DEFAULT_DURATION_MS);
    }

    /**
     * Get the message.
     *
     * @return The message
     */
    public String message() {
        return message;
    }

    /**
     * Get the kind.
     *
     * @return The kind
     */
    public Kind kind() {
        return kind;
    }

    /**
     * Get the duration.
     *
     * @return The duration in milliseconds
     */
    public long durationMillis() {
        return durationMillis;
    }

    /**
     * Check if the toast has expired.
     *
     * @return true if expired
     */
    public boolean expired() {
        return expired(System.nanoTime());
    }

    /**
     * Check if the toast has expired.
     *
     * @param nowNanos The current time in nanoseconds
     * @return true if expired
     */
    public boolean expired(final long nowNanos) {
        return durationMillis > 0
                && shownNanos != 0L
                && nowNanos - shownNanos >= durationMillis * 1_000_000L;
    }

    /**
     * Get the ANSI style code for the kind.
     *
     * @return The style code
     */
    public String style() {
        final Theme theme = Theme.current();
        return switch (kind) {
            case SUCCESS -> theme.success();
            case WARNING -> theme.warning();
            case ERROR -> theme.error();
            case INFO -> theme.info();
        };
    }
}
