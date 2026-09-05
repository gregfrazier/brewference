package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Overlay;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.list.ListModel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A choice overlay for selecting one or more items from a list.
 * @param <T> The type of items
 */
public final class ChoiceOverlay<T> implements Overlay {
    private final String title;
    private final ListModel<T> model;
    private final Function<T, String> label;
    private final Consumer<T> onPick;
    private final Consumer<List<T>> onMulti;
    private final boolean multi;
    private UiHost host;
    private boolean invalidated = true;
    private boolean finished;

    /**
     * Construct a choice overlay.
     *
     * @param title The overlay title
     * @param items The initial items
     * @param label Function to extract labels from items
     * @param onPick Callback for single selection
     * @param onMulti Callback for multi-selection
     * @param multi Whether to enable multi-selection
     */
    private ChoiceOverlay(final String title,
                          final Collection<T> items,
                          final Function<T, String> label,
                          final Consumer<T> onPick,
                          final Consumer<List<T>> onMulti,
                          final boolean multi) {
        this.title = title == null || title.isBlank() ? "Choose" : title;
        this.label = label == null ? String::valueOf : label;
        this.model = new ListModel<>(items, this.label);
        this.onPick = onPick == null ? ignored -> {} : onPick;
        this.onMulti = onMulti == null ? ignored -> {} : onMulti;
        this.multi = multi;
    }

    /**
     * Create a string choice overlay.
     *
     * @param title The overlay title
     * @param items The list of items
     * @param onPick Callback for selection
     * @return The ChoiceOverlay
     */
    public static ChoiceOverlay<String> of(final String title,
                                           final List<String> items,
                                           final Consumer<String> onPick) {
        return of(title, items, item -> item, onPick);
    }

    /**
     * Create a choice overlay.
     *
     * @param title The overlay title
     * @param items The list of items
     * @param label Function to extract labels from items
     * @param onPick Callback for selection
     * @return The ChoiceOverlay
     */
    public static <T> ChoiceOverlay<T> of(final String title,
                                          final List<T> items,
                                          final Function<T, String> label,
                                          final Consumer<T> onPick) {
        return new ChoiceOverlay<>(title, items, label, onPick, null, false);
    }

    /**
     * Create a multi-selection choice overlay.
     *
     * @param title The overlay title
     * @param items The list of items
     * @param label Function to extract labels from items
     * @param onPick Callback for selection
     * @return The ChoiceOverlay
     */
    public static <T> ChoiceOverlay<T> multi(final String title,
                                             final List<T> items,
                                             final Function<T, String> label,
                                             final Consumer<List<T>> onPick) {
        return new ChoiceOverlay<>(title, items, label, null, onPick, true);
    }

    /**
     * Called when the choice overlay opens.
     *
     * @param host The UI host
     */
    @Override
    public void onOpen(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Render the choice overlay body.
     *
     * @param ctx The view context
     * @param underlyingBody The underlying body
     * @return The rendered overlay body
     */
    @Override
    public String renderBody(final ViewContext ctx, final String underlyingBody) {
        invalidated = false;
        final List<String> rows = new ArrayList<>();
        final List<T> visible = model.visibleItems();
        if (visible.isEmpty()) {
            rows.add("No choices.");
        } else {
            for (int i = 0; i < visible.size(); i++) {
                final T item = visible.get(i);
                final String mark = i == model.getCursor() ? ">" : " ";
                final String selected = multi && model.isSelected(item) ? "*" : " ";
                rows.add(mark + selected + " " + label.apply(item));
            }
        }
        rows.add("");
        rows.add(multi ? "[Space] toggle   [Enter] accept   [Esc] cancel"
                : "[Enter] choose   [Esc] cancel");
        return ModalBox.paint(ctx, underlyingBody, title, rows);
    }

    /**
     * Handle events for the choice overlay.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (finished || !(msg instanceof KeyPressed(String key))) {
            return;
        }
        if ("ESC".equals(key)) {
            close();
            return;
        }
        if ("ENTER".equals(key)) {
            finish();
            return;
        }
        if (multi && (" ".equals(key) || "SPACE".equals(key))) {
            model.toggleSelected();
        } else if ("UP".equals(key)) {
            model.moveUp();
        } else if ("DOWN".equals(key)) {
            model.moveDown();
        } else if ("PAGE_UP".equals(key)) {
            model.movePageUp();
        } else if ("PAGE_DOWN".equals(key)) {
            model.movePageDown();
        } else if ("HOME".equals(key)) {
            model.moveFirst();
        } else if ("END".equals(key)) {
            model.moveLast();
        }
        invalidated = true;
    }

    /**
     * Check if the choice overlay should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Finish the choice overlay.
     */
    private void finish() {
        finished = true;
        close();
        if (multi) {
            onMulti.accept(model.selectedItems());
        } else {
            model.selectEntry().ifPresent(onPick);
        }
    }

    /**
     * Close the choice overlay.
     */
    private void close() {
        finished = true;
        if (host != null) {
            host.popOverlay();
        }
    }
}
