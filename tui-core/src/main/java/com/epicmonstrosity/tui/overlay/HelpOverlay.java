package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Overlay;
import com.epicmonstrosity.tui.Screen;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.screen.KeyBinding;

import java.util.ArrayList;
import java.util.List;

/**
 * A help overlay showing key bindings for a screen.
 */
public final class HelpOverlay implements Overlay {
    private final List<String> rows;
    private UiHost host;
    private boolean invalidated = true;

    /**
     * Construct a help overlay.
     *
     * @param rows The help text rows
     */
    private HelpOverlay(final List<String> rows) {
        this.rows = List.copyOf(rows);
    }

    /**
     * Create a help overlay from a screen's key bindings.
     *
     * @param screen The screen to get help for
     * @return The HelpOverlay
     */
    public static HelpOverlay from(final Screen screen) {
        final List<String> rows = new ArrayList<>();
        if (screen != null) {
            for (KeyBinding binding : screen.keyBindings()) {
                rows.add(binding.formatted());
            }
        }
        if (rows.isEmpty()) {
            rows.add("No key bindings on this screen.");
        }
        rows.add("");
        rows.add("[Esc] close help");
        return new HelpOverlay(rows);
    }

    /**
     * Called when the help overlay opens.
     *
     * @param host The UI host
     */
    @Override
    public void onOpen(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Render the help overlay body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @return The rendered overlay body
     */
    @Override
    public String renderBody(final ViewContext ctx, final String underlyingBody) {
        invalidated = false;
        return ModalBox.paint(ctx, underlyingBody, "Help", rows);
    }

    /**
     * Handle events for the help overlay.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof KeyPressed(String key) && "ESC".equals(key) && host != null) {
            host.popOverlay();
        }
    }

    /**
     * Check if the help overlay should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

}
