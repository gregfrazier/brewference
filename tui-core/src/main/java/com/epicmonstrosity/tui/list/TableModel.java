package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.ViewContext;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

/**
 * Snapshot of a {@link Collection} as a random-access, sortable table.
 * Input may be a {@link List}, {@link java.util.Deque}, or any other collection;
 * rows are copied into an {@link ArrayList}. Use {@link #add} / {@link #addAll}
 * for live growth (scan results, queues) instead of mutating the original.
 */
public final class TableModel<T> implements Navigable<T> {
    private final List<T> items = new ArrayList<>();
    private final List<TableColumn<T>> columns;
    private int sortColumn;
    private boolean sortAscending = true;
    private boolean sorted;
    private int cursor;
    private int scrollOffset;
    private String statusMessage = "";

    /**
     * Create a table model.
     *
     * @param columns The table columns
     * @param items The initial items
     */
    public TableModel(final List<TableColumn<T>> columns, final Collection<? extends T> items) {
        this.columns = columns == null ? List.of() : List.copyOf(columns);
        this.sortColumn = firstSortable();
        setItems(items);
    }

    /**
     * Get the table columns.
     *
     * @return The columns
     */
    public List<TableColumn<T>> columns() {
        return columns;
    }

    /**
     * Get the items.
     *
     * @return The items
     */
    public List<T> items() {
        return List.copyOf(items);
    }

    /**
     * Get the number of items.
     *
     * @return The size
     */
    public int size() {
        return items.size();
    }

    /**
     * Set the items.
     *
     * @param source The new items
     */
    public void setItems(final Collection<? extends T> source) {
        items.clear();
        if (source != null) {
            items.addAll(source);
        }
        if (sorted) {
            applySort();
        } else {
            clampCursor();
        }
    }

    /**
     * Add an item.
     *
     * @param item The item to add
     */
    public void add(final T item) {
        if (item == null) {
            return;
        }
        items.add(item);
        if (sorted) {
            applySort();
        } else {
            clampCursor();
        }
    }

    /**
     * Add multiple items.
     *
     * @param more The items to add
     */
    public void addAll(final Collection<? extends T> more) {
        if (more == null || more.isEmpty()) {
            return;
        }
        for (T item : more) {
            if (item != null) {
                items.add(item);
            }
        }
        if (sorted) {
            applySort();
        } else {
            clampCursor();
        }
    }

    /**
     * Clear all items and reset state.
     */
    public void clear() {
        items.clear();
        cursor = 0;
        scrollOffset = 0;
        sorted = false;
        sortAscending = true;
        sortColumn = firstSortable();
        statusMessage = "";
    }

    /**
     * Check if the table is sortable.
     *
     * @return true if sortable
     */
    public boolean sortable() {
        return !sortableIndexes().isEmpty();
    }

    /**
     * Get the sort column.
     *
     * @return The sort column index
     */
    public int sortColumn() {
        return sortColumn;
    }

    /**
     * Check if sorting is ascending.
     *
     * @return true if ascending
     */
    public boolean sortAscending() {
        return sortAscending;
    }

    /**
     * Check if the table is sorted.
     *
     * @return true if sorted
     */
    public boolean sorted() {
        return sorted;
    }

    /**
     * Get the status message.
     *
     * @return The status
     */
    public String statusMessage() {
        return statusMessage;
    }

    /**
     * Set the status message.
     *
     * @param statusMessage The status
     */
    public void setStatusMessage(final String statusMessage) {
        this.statusMessage = statusMessage == null ? "" : statusMessage;
    }

    /**
     * Cycle through sort columns.
     */
    public void cycleSort() {
        final List<Integer> sortable = sortableIndexes();
        if (sortable.isEmpty()) {
            return;
        }
        final int pos = sortable.indexOf(sortColumn);
        final int next = pos < 0 ? 0 : (pos + 1) % sortable.size();
        if (next == 0 && pos >= 0) {
            sortAscending = !sortAscending;
        }
        sortColumn = sortable.get(next);
        applySort();
        final TableColumn<T> column = columns.get(sortColumn);
        statusMessage = "Sorted by " + column.header() + " (" + (sortAscending ? "asc" : "desc") + ")";
    }

    /**
     * Sort by a specific column.
     *
     * @param columnIndex The column index
     * @param ascending Whether to sort ascending
     */
    public void sortBy(final int columnIndex, final boolean ascending) {
        if (columnIndex < 0 || columnIndex >= columns.size() || !columns.get(columnIndex).sortable()) {
            return;
        }
        sortColumn = columnIndex;
        sortAscending = ascending;
        applySort();
        statusMessage = "Sorted by " + columns.get(sortColumn).header() + " (" + (sortAscending ? "asc" : "desc") + ")";
    }

    /**
     * Get items matching a predicate.
     *
     * @param predicate The predicate
     * @return Matching items
     */
    public List<T> matching(final java.util.function.Predicate<T> predicate) {
        if (predicate == null) {
            return items();
        }
        return items.stream().filter(predicate).toList();
    }

    /**
     * Get the visible window of items.
     *
     * @param viewportHeight The viewport height
     * @return The visible items
     */
    public List<T> visibleWindow(final int viewportHeight) {
        if (items.isEmpty() || viewportHeight <= 0) {
            scrollOffset = 0;
            return List.of();
        }
        scrollOffset = ViewContext.scrollOffset(getCursor(), scrollOffset, viewportHeight);
        final int from = Math.min(scrollOffset, Math.max(0, items.size() - 1));
        final int to = Math.min(items.size(), from + viewportHeight);
        return items.subList(from, to);
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
     * Get the number of entries.
     *
     * @return The size
     */
    @Override
    public int getEntriesSize() {
        return items.size();
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
        if (index < 0 || index >= items.size()) {
            return null;
        }
        return items.get(index);
    }

    /**
     * Apply sorting.
     */
    private void applySort() {
        if (sortColumn < 0 || sortColumn >= columns.size()) {
            return;
        }
        final TableColumn<T> column = columns.get(sortColumn);
        if (column.sort() == null) {
            return;
        }
        final T selected = selectEntry().orElse(null);
        Comparator<T> cmp = column.sort();
        if (!sortAscending) {
            cmp = cmp.reversed();
        }
        items.sort(cmp);
        sorted = true;
        if (selected != null) {
            final int index = items.indexOf(selected);
            if (index >= 0) {
                cursor = index;
            }
        }
        clampCursor();
    }

    /**
     * Get the indexes of sortable columns.
     *
     * @return The sortable indexes
     */
    private List<Integer> sortableIndexes() {
        final List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < columns.size(); i++) {
            if (columns.get(i).sortable()) {
                indexes.add(i);
            }
        }
        return indexes;
    }

    /**
     * Get the first sortable column index.
     *
     * @return The first sortable index, or -1 if none
     */
    private int firstSortable() {
        for (int i = 0; i < columns.size(); i++) {
            if (columns.get(i).sortable()) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Clamp the cursor to valid range.
     */
    private void clampCursor() {
        if (items.isEmpty()) {
            cursor = 0;
            scrollOffset = 0;
            return;
        }
        cursor = Math.max(0, Math.min(cursor, items.size() - 1));
    }
}
