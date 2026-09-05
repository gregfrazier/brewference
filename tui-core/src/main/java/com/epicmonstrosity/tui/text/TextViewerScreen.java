package com.epicmonstrosity.tui.text;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.TerminalResized;
import com.epicmonstrosity.tui.Tick;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.util.List;

/**
 * A screen for viewing text files.
 */
public final class TextViewerScreen extends ScreenTemplate {
    private final String title;
    private final TextViewerModel model;
    private UiHost host;
    private boolean invalidated = true;
    private boolean composingSearch;
    private int cols = 80;
    private int viewportHeight = 10;

    /**
     * Create a text viewer screen.
     *
     * @param title The screen title
     * @param model The text viewer model
     */
    public TextViewerScreen(final String title, final TextViewerModel model) {
        this.title = title == null ? "Text" : title;
        this.model = model;
    }

    /**
     * Build the text viewer layout.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        cols = Math.max(1, ctx.terminalCols() - 2);
        viewportHeight = Math.max(1, ctx.bodyRows() - 2);
        final List<String> window = model.visibleWindow(cols, viewportHeight);
        final var body = new StringBuilder();
        final int from = model.scrollOffset();
        for (int i = 0; i < window.size(); i++) {
            final int lineIndex = from + i;
            final String line = window.get(i);
            if (lineIndex == model.matchIndex()) {
                body.append(YELLOW).append(line).append(RESET).append('\n');
            } else {
                body.append(line).append('\n');
            }
        }
        final int total = model.totalWrapped(cols);
        if (total > viewportHeight) {
            body.append(DIM)
                    .append(scrollIndicator(model.scrollOffset(), viewportHeight, total))
                    .append(RESET)
                    .append('\n');
        }
        if (composingSearch || !model.query().isEmpty()) {
            body.append(DIM).append("  /").append(RESET).append(model.query());
            if (composingSearch) {
                body.append(BG_SEL).append(' ').append(RESET);
            }
            body.append('\n');
        }
        return new ScreenLayout(
                headerBordered(title + followBadge(), "", ctx.terminalCols() - 2),
                body.toString(),
                "",
                footerError(host == null ? "" : host.errorBanner())
        );
    }

    /**
     * Get the key bindings for the text viewer.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        return List.of(
                KeyBinding.primary("UP", "↑", "up", () -> model.moveBy(-1, cols, viewportHeight)),
                KeyBinding.primary("DOWN", "↓", "down", () -> model.moveBy(1, cols, viewportHeight)),
                KeyBinding.primary("PAGE_UP", "⇑", "pg up", () -> model.pageUp(cols, viewportHeight)),
                KeyBinding.primary("PAGE_DOWN", "⇓", "pg down", () -> model.pageDown(cols, viewportHeight)),
                KeyBinding.secondary("/", "/", "search", this::startSearch),
                KeyBinding.secondary("n", "n", "next", () -> model.findNext(cols, viewportHeight)),
                KeyBinding.secondary("N", "N", "prev", () -> model.findPrevious(cols, viewportHeight)),
                KeyBinding.secondary("HOME", "↖", "home", model::home),
                KeyBinding.secondary("END", "↘", "end", () -> model.end(cols, viewportHeight)),
                KeyBinding.secondary("F", "F", "follow", this::follow),
                KeyBinding.secondary("ESC", "Esc", "back", this::back)
        );
    }

    /**
     * Called when the text viewer enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the text viewer exits.
     */
    @Override
    public void onExit() {
    }

    /**
     * Handle UI events for the text viewer.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof TerminalResized || msg instanceof Tick) {
            model.poll();
            invalidated = true;
            return;
        }
        if (msg instanceof Pasted(String text) && composingSearch) {
            model.typeQuery(text.replace("\n", ""));
            invalidated = true;
            return;
        }
        if (msg instanceof KeyPressed(String key)) {
            if (composingSearch) {
                handleSearchKey(key);
            } else if ("/".equals(key)) {
                startSearch();
            } else if ("f".equals(key) || "F".equals(key)) {
                follow();
            } else {
                handleInput(key);
            }
            invalidated = true;
        }
    }

    /**
     * Check if the text viewer should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        if (model.poll()) {
            invalidated = true;
        }
        return invalidated;
    }

    /**
     * Handle search key input.
     *
     * @param key The key pressed
     */
    private void handleSearchKey(final String key) {
        if ("ENTER".equals(key)) {
            composingSearch = false;
            model.findNext(cols, viewportHeight);
            return;
        }
        if ("ESC".equals(key)) {
            composingSearch = false;
            model.clearSearch();
            return;
        }
        if ("BACKSPACE".equals(key)) {
            model.backspaceQuery();
            return;
        }
        if (key.length() == 1 && key.charAt(0) >= 32) {
            model.typeQuery(key);
        }
    }

    /**
     * Toggle following the tail.
     */
    private void follow() {
        model.followTail(true);
        model.end(cols, viewportHeight);
    }

    /**
     * Get the follow badge.
     *
     * @return The badge string
     */
    private String followBadge() {
        if (!model.followable()) {
            return "";
        }
        return model.following() ? "  FOLLOW" : "  PAUSED";
    }

    /**
     * Start a search.
     */
    private void startSearch() {
        composingSearch = true;
        model.setQuery("");
    }

    /**
     * Go back to the previous screen.
     */
    private void back() {
        if (host != null) {
            host.navigation().pop();
        }
    }
}
