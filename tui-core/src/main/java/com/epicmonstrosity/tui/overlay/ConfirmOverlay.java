package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Overlay;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A confirm overlay for yes/no or type-to-confirm dialogs.
 */
public final class ConfirmOverlay implements Overlay {
    private final String title;
    private final String message;
    private final String expected;
    private final Consumer<Boolean> onResult;
    private final StringBuilder typed = new StringBuilder();
    private UiHost host;
    private boolean invalidated = true;
    private boolean finished;

    /**
     * Construct a confirm overlay.
     *
     * @param title The overlay title
     * @param message The message to display
     * @param expected The expected input for type-to-confirm (null for yes/no)
     * @param onResult Callback for the result
     */
    private ConfirmOverlay(final String title,
                           final String message,
                           final String expected,
                           final Consumer<Boolean> onResult) {
        this.title = title == null || title.isBlank() ? "Confirm" : title;
        this.message = message == null ? "" : message;
        this.expected = expected;
        this.onResult = onResult == null ? ignored -> {} : onResult;
    }

    /**
     * Create a yes/no confirm overlay.
     *
     * @param message The message to display
     * @param onResult Callback for the result
     * @return The ConfirmOverlay
     */
    public static ConfirmOverlay yesNo(final String message, final Consumer<Boolean> onResult) {
        return new ConfirmOverlay("Confirm", message, null, onResult);
    }

    /**
     * Create a type-to-confirm overlay.
     *
     * @param message The message to display
     * @param expected The expected input
     * @param onResult Callback for the result
     * @return The ConfirmOverlay
     */
    public static ConfirmOverlay typeToConfirm(final String message,
                                               final String expected,
                                               final Consumer<Boolean> onResult) {
        return new ConfirmOverlay("Confirm", message, expected == null ? "" : expected, onResult);
    }

    /**
     * Called when the confirm overlay opens.
     *
     * @param host The UI host
     */
    @Override
    public void onOpen(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Render the confirm overlay body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @return The rendered overlay body
     */
    @Override
    public String renderBody(final ViewContext ctx, final String underlyingBody) {
        invalidated = false;
        final List<String> rows = new ArrayList<>();
        if (!message.isBlank()) {
            rows.add(message);
            rows.add("");
        }
        if (expected != null) {
            rows.add("Type " + expected + ":");
            rows.add("> " + typed);
            rows.add("");
            rows.add("[Enter] confirm   [Esc] cancel");
        } else {
            rows.add("[Y] yes   [N] no   [Esc] cancel");
        }
        return ModalBox.paint(ctx, underlyingBody, title, rows);
    }

    /**
     * Handle events for the confirm overlay.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (finished) {
            return;
        }
        if (msg instanceof Pasted(String text) && expected != null) {
            typed.append(text.replace("\n", ""));
            invalidated = true;
            return;
        }
        if (!(msg instanceof KeyPressed(String key))) {
            return;
        }
        if (expected != null) {
            handleTyped(key);
        } else {
            handleYesNo(key);
        }
        invalidated = true;
    }

    /**
     * Check if the confirm overlay should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Handle yes/no input.
     *
     * @param key The key pressed
     */
    private void handleYesNo(final String key) {
        if ("y".equalsIgnoreCase(key) || "ENTER".equals(key)) {
            finish(true);
        } else if ("n".equalsIgnoreCase(key) || "ESC".equals(key)) {
            finish(false);
        }
    }

    /**
     * Handle typed input.
     *
     * @param key The key pressed
     */
    private void handleTyped(final String key) {
        if ("ESC".equals(key)) {
            finish(false);
            return;
        }
        if ("ENTER".equals(key)) {
            if (expected.equals(typed.toString())) {
                finish(true);
            }
            return;
        }
        if ("BACKSPACE".equals(key)) {
            if (!typed.isEmpty()) {
                typed.deleteCharAt(typed.length() - 1);
            }
            return;
        }
        if (isPrintable(key)) {
            typed.append(key);
        }
    }

    /**
     * Finish the confirm overlay.
     *
     * @param confirmed Whether the confirmation was accepted
     */
    private void finish(final boolean confirmed) {
        finished = true;
        if (host != null) {
            host.popOverlay();
        }
        onResult.accept(confirmed);
    }

    /**
     * Check if a key is printable.
     *
     * @param key The key to check
     * @return true if printable
     */
    private static boolean isPrintable(final String key) {
        return key.length() == 1 && key.charAt(0) >= 32;
    }
}
