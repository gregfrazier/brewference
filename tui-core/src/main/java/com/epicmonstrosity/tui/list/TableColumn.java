package com.epicmonstrosity.tui.list;

import java.util.Comparator;
import java.util.function.Function;

public record TableColumn<T>(
        String header,
        int width,
        Function<T, String> value,
        Comparator<T> sort,
        Function<T, String> style
) {
    /**
     * Create a table column.
     *
     * @param header The column header
     * @param width The column width
     * @param value Function to extract value from items
     * @return The TableColumn
     */
    public static <T> TableColumn<T> of(final String header, final int width, final Function<T, String> value) {
        return new TableColumn<>(header, width, value, null, null);
    }

    /**
     * Make the column sortable.
     *
     * @param sort The comparator
     * @return A new TableColumn with sorting enabled
     */
    public TableColumn<T> sortable(final Comparator<T> sort) {
        return new TableColumn<>(header, width, value, sort, style);
    }

    /**
     * Set a style function for the column.
     *
     * @param style Function to extract style from items
     * @return A new TableColumn with styling
     */
    public TableColumn<T> style(final Function<T, String> style) {
        return new TableColumn<>(header, width, value, sort, style);
    }

    /**
     * Check if the column is sortable.
     *
     * @return true if sortable
     */
    public boolean sortable() {
        return sort != null;
    }
}
