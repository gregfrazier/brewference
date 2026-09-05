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
 * A prompt overlay for typing text input.
 */
public final class PromptOverlay implements Overlay {
    private final String title;
    private final StringBuilder buffer;
    private final Consumer<String> onSubmit;
    private UiHost host;
    private boolean invalidated = true;
    private boolean finished;

    /**
     * Construct a prompt overlay.
     *
     * @param title The overlay title
     * @param initial The initial text
     * @param onSubmit Callback for the submitted text
     */
    private PromptOverlay(final String title, final String initial, final Consumer<String> onSubmit) {
        this.title = title == null || title.isBlank() ? "Prompt" : title;
        this.buffer = new StringBuilder(initial == null ? "" : initial);
        this.onSubmit = onSubmit == null ? ignored -> {} : onSubmit;
    }

    /**
     * Create a prompt overlay.
     *
     * @param title The overlay title
     * @param initial The initial text
     * @param onSubmit Callback for the submitted text
     * @return The PromptOverlay
     */
    public static PromptOverlay ask(final String title, final String initial, final Consumer<String> onSubmit) {
        return new PromptOverlay(title, initial, onSubmit);
    }

    /**
     * Called when the prompt overlay opens.
     *
     * @param host The UI host
     */
    @Override
    public void onOpen(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Render the prompt overlay body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @return The rendered overlay body
     */
    @Override
    public String renderBody(final ViewContext ctx, final String underlyingBody) {
        invalidated = false;
        final List<String> rows = new ArrayList<>();
        rows.add("> " + buffer);
        rows.add("");
        rows.add("[Enter] submit   [Esc] cancel");
        return ModalBox.paint(ctx, underlyingBody, title, rows);
    }

    /**
     * Handle events for the prompt overlay.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (finished) {
            return;
        }
        if (msg instanceof Pasted(String text)) {
            buffer.append(text.replace("\n", ""));
            invalidated = true;
            return;
        }
        if (!(msg instanceof KeyPressed(String key))) {
            return;
        }
        if ("ESC".equals(key)) {
            close();
            return;
        }
        if ("ENTER".equals(key)) {
            finished = true;
            close();
            onSubmit.accept(buffer.toString());
            return;
        }
        if ("BACKSPACE".equals(key)) {
            if (!buffer.isEmpty()) {
                buffer.deleteCharAt(buffer.length() - 1);
            }
        } else if (isPrintable(key)) {
            buffer.append(key);
        }
        invalidated = true;
    }

    /**
     * Check if the prompt overlay should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Close the prompt overlay.
     */
    private void close() {
        invalidated = true;
        if (host != null) {
            host.popOverlay();
        }
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
