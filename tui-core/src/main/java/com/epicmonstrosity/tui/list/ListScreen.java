package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A list screen for displaying and selecting items.
 * @param <T> The type of items
 */
public final class ListScreen<T> extends ScreenTemplate {
    private final String title;
    private final ListModel<T> model;
    private final Function<T, String> renderer;
    private final Consumer<T> onSelect;
    private final Consumer<List<T>> onMultiSelect;
    private final boolean multiSelect;
    private final boolean filterable;
    private UiHost host;
    private boolean invalidated = true;
    private int viewportHeight = 10;

    /**
     * Create a list screen.
     *
     * @param title The screen title
     * @param items The initial items
     * @param renderer Function to render items
     * @param onSelect Callback for selection
     * @param onMultiSelect Callback for multi-selection
     * @param multiSelect Whether to enable multi-selection
     * @param filterable Whether to enable filtering
     */
    private ListScreen(final String title,
                       final Collection<T> items,
                       final Function<T, String> renderer,
                       final Consumer<T> onSelect,
                       final Consumer<List<T>> onMultiSelect,
                       final boolean multiSelect,
                       final boolean filterable) {
        this.title = title == null ? "List" : title;
        this.renderer = renderer == null ? String::valueOf : renderer;
        this.model = new ListModel<>(items, this.renderer);
        this.onSelect = onSelect == null ? ignored -> {} : onSelect;
        this.onMultiSelect = onMultiSelect == null ? ignored -> {} : onMultiSelect;
        this.multiSelect = multiSelect;
        this.filterable = filterable;
    }

    /**
     * Create a single-selection list screen.
     *
     * @param title The screen title
     * @param items The initial items
     * @param renderer Function to render items
     * @param onSelect Callback for selection
     * @return The ListScreen
     */
    public static <T> ListScreen<T> single(final String title,
                                           final Collection<T> items,
                                           final Function<T, String> renderer,
                                           final Consumer<T> onSelect) {
        return new ListScreen<>(title, items, renderer, onSelect, null, false, true);
    }

    /**
     * Create a multi-selection list screen.
     *
     * @param title The screen title
     * @param items The initial items
     * @param renderer Function to render items
     * @param onSelect Callback for selection
     * @return The ListScreen
     */
    public static <T> ListScreen<T> multi(final String title,
                                          final Collection<T> items,
                                          final Function<T, String> renderer,
                                          final Consumer<List<T>> onSelect) {
        return new ListScreen<>(title, items, renderer, null, onSelect, true, true);
    }

    /**
     * Build the list layout.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        viewportHeight = Math.max(1, ctx.bodyRows() - (filterable ? 2 : 1));
        final var body = new StringBuilder();
        if (filterable) {
            body.append(DIM).append("  Filter: ").append(RESET).append(model.filter()).append('\n');
        }
        final List<T> window = model.visibleWindow(viewportHeight);
        final int from = model.scrollOffset();
        for (int i = 0; i < window.size(); i++) {
            final T item = window.get(i);
            final int index = from + i;
            final String mark = multiSelect && model.isSelected(item) ? "*" : " ";
            configField(body, model.getCursor(), index, mark, renderer.apply(item));
        }
        if (model.getEntriesSize() > viewportHeight) {
            body.append(DIM)
                    .append(scrollIndicator(model.scrollOffset(), viewportHeight, model.getEntriesSize()))
                    .append(RESET)
                    .append('\n');
        }
        return new ScreenLayout(
                headerBordered(title, "", ctx.terminalCols() - 2),
                body.toString(),
                "",
                footerError(host == null ? "" : host.errorBanner())
        );
    }

    /**
     * Get the key bindings for the list screen.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        final List<KeyBinding> bindings = new ArrayList<>();
        bindings.add(KeyBinding.primary("UP", "↑", "up", model::moveUp));
        bindings.add(KeyBinding.primary("DOWN", "↓", "down", model::moveDown));
        bindings.add(KeyBinding.primary("ENTER", "Enter", multiSelect ? "accept" : "select", this::accept));
        if (multiSelect) {
            bindings.add(KeyBinding.primary(" ", "Space", "toggle", model::toggleSelected));
        }
        bindings.add(KeyBinding.secondary("PAGE_UP", "⇑", "pg up", () -> model.moveCursor(-viewportHeight)));
        bindings.add(KeyBinding.secondary("PAGE_DOWN", "⇓", "pg down", () -> model.moveCursor(viewportHeight)));
        bindings.add(KeyBinding.secondary("HOME", "↖", "home", model::moveFirst));
        bindings.add(KeyBinding.secondary("END", "↘", "end", model::moveLast));
        bindings.add(KeyBinding.secondary("ESC", "Esc", "back", this::back));
        return bindings;
    }

    /**
     * Called when the list screen enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the list screen exits.
     */
    @Override
    public void onExit() {
    }

    /**
     * Handle UI events for the list screen.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof Pasted(String text) && filterable) {
            model.typeFilter(text.replace("\n", ""));
            invalidated = true;
            return;
        }
        if (msg instanceof KeyPressed(String key)) {
            if (filterable && isPrintable(key) && !" ".equals(key)) {
                model.typeFilter(key);
            } else if (filterable && "BACKSPACE".equals(key)) {
                model.backspaceFilter();
            } else if ("ESC".equals(key) && filterable && !model.filter().isEmpty()) {
                model.setFilter("");
            } else {
                handleInput(key);
            }
            invalidated = true;
        }
    }

    /**
     * Check if the list screen should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Accept the selection.
     */
    private void accept() {
        if (multiSelect) {
            onMultiSelect.accept(model.selectedItems());
        } else {
            model.selectEntry().ifPresent(onSelect);
        }
        if (host != null) {
            host.navigation().pop();
        }
    }

    /**
     * Go back to the previous screen.
     */
    private void back() {
        if (host != null) {
            host.navigation().pop();
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
