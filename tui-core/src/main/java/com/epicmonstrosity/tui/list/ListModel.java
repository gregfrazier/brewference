package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.ViewContext;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Function;

/**
 * A list model for managing items with filtering and selection.
 * @param <T> The type of items
 */
public final class ListModel<T> implements Navigable<T> {
    private final List<T> items = new ArrayList<>();
    private final Set<T> selected = new LinkedHashSet<>();
    private final Function<T, String> filterText;
    private String filter = "";
    private int cursor;
    private int scrollOffset;

    /**
     * Create a list model with default string filter.
     *
     * @param items The initial items
     */
    public ListModel(final Collection<T> items) {
        this(items, String::valueOf);
    }

    /**
     * Create a list model with a custom filter function.
     *
     * @param items The initial items
     * @param filterText Function to convert items to strings for filtering
     */
    public ListModel(final Collection<T> items, final Function<T, String> filterText) {
        this.filterText = filterText == null ? String::valueOf : filterText;
        setItems(items);
    }

    /**
     * Set the items and clear selection.
     *
     * @param items The new items
     */
    public void setItems(final Collection<T> items) {
        this.items.clear();
        if (items != null) {
            this.items.addAll(items);
        }
        selected.retainAll(this.items);
        clampCursor();
    }

    /**
     * Get the filter text.
     *
     * @return The filter
     */
    public String filter() {
        return filter;
    }

    /**
     * Set the filter text.
     *
     * @param filter The filter
     */
    public void setFilter(final String filter) {
        this.filter = filter == null ? "" : filter;
        cursor = 0;
        scrollOffset = 0;
        clampCursor();
    }

    /**
     * Type a character into the filter.
     *
     * @param text The character
     */
    public void typeFilter(final String text) {
        setFilter(filter + text);
    }

    /**
     * Delete the last character from the filter.
     */
    public void backspaceFilter() {
        if (!filter.isEmpty()) {
            setFilter(filter.substring(0, filter.length() - 1));
        }
    }

    /**
     * Get the visible (filtered) items.
     *
     * @return The visible items
     */
    public List<T> visibleItems() {
        if (filter.isBlank()) {
            return List.copyOf(items);
        }
        final String needle = filter.toLowerCase(Locale.ROOT);
        return items.stream()
                .filter(item -> filterText.apply(item).toLowerCase(Locale.ROOT).contains(needle))
                .toList();
    }

    /**
     * Get the visible window of items.
     *
     * @param viewportHeight The viewport height
     * @return The visible items
     */
    public List<T> visibleWindow(final int viewportHeight) {
        final List<T> visible = visibleItems();
        if (visible.isEmpty() || viewportHeight <= 0) {
            scrollOffset = 0;
            return List.of();
        }
        scrollOffset = ViewContext.scrollOffset(getCursor(), scrollOffset, viewportHeight);
        final int from = Math.min(scrollOffset, Math.max(0, visible.size() - 1));
        final int to = Math.min(visible.size(), from + viewportHeight);
        return visible.subList(from, to);
    }

    /**
     * Toggle selection of the current item.
     */
    public void toggleSelected() {
        selectEntry().ifPresent(item -> {
            if (!selected.add(item)) {
                selected.remove(item);
            }
        });
    }

    /**
     * Check if an item is selected.
     *
     * @param item The item to check
     * @return true if selected
     */
    public boolean isSelected(final T item) {
        return selected.contains(item);
    }

    /**
     * Get all selected items.
     *
     * @return The selected items
     */
    public List<T> selectedItems() {
        return items.stream().filter(selected::contains).toList();
    }

    /**
     * Clear all selections.
     */
    public void clearSelection() {
        selected.clear();
    }

    /**
     * Get the scroll offset.
     *
     * @return The scroll offset
     */
    public int scrollOffset() {
        return scrollOffset;
    }

    /**
     * Get the number of visible entries.
     *
     * @return The size
     */
    @Override
    public int getEntriesSize() {
        return visibleItems().size();
    }

    /**
     * Get the cursor position.
     *
     * @return The cursor
     */
    @Override
    public int getCursor() {
        return cursor;
    }

    /**
     * Set the cursor position.
     *
     * @param cursor The cursor
     */
    @Override
    public void setCursor(final int cursor) {
        this.cursor = cursor;
        clampCursor();
    }

    /**
     * Get an entry by index.
     *
     * @param index The index
     * @return The entry, or null if out of bounds
     */
    @Override
    public T getEntry(final int index) {
        final List<T> visible = visibleItems();
        if (index < 0 || index >= visible.size()) {
            return null;
        }
        return visible.get(index);
    }

    /**
     * Clamp the cursor to valid range.
     */
    private void clampCursor() {
        final int size = visibleItems().size();
        if (size == 0) {
            cursor = 0;
            scrollOffset = 0;
            return;
        }
        cursor = Math.max(0, Math.min(cursor, size - 1));
    }
}
