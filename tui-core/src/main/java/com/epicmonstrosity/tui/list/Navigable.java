package com.epicmonstrosity.tui.list;

import java.util.Optional;

/**
 * Interface for navigable list/table models.
 * @param <T> The type of entries
 */
public interface Navigable<T> {
    int getEntriesSize();
    int getCursor();
    void setCursor(int cursor);
    T getEntry(int index);

    /**
     * Move the cursor by a delta.
     *
     * @param delta The delta to move
     */
    default void moveCursor(final int delta) {
        if (getEntriesSize() > 0)
            setCursor(clamp(getCursor() + delta, 0, getEntriesSize() - 1));
    }

    /**
     * Move the cursor up.
     */
    default void moveUp() {
        moveCursor(-1);
    }

    /**
     * Move the cursor to the first entry.
     */
    default void moveFirst() {
        setCursor(0);
    }

    /**
     * Move the cursor to the last entry.
     */
    default void moveLast() {
        if (getEntriesSize() > 0)
            setCursor(getEntriesSize() - 1);
    }

    /**
     * Move the cursor up by a page.
     */
    default void movePageUp() {
        moveCursor(-10);
    }

    /**
     * Move the cursor down by a page.
     */
    default void movePageDown() {
        moveCursor(10);
    }

    /**
     * Select (get) the current entry.
     *
     * @return Optional of the entry
     */
    default Optional<T> selectEntry() {
        if (getEntriesSize() == 0) return Optional.empty();
        final var entry = getEntry(getCursor());
        if (entry == null) return Optional.empty();
        return Optional.of(entry);
    }

    /**
     * Move the cursor down.
     */
    default void moveDown() {
        moveCursor(1);
    }

    /**
     * Clamp a value to a range.
     *
     * @param val The value to clamp
     * @param min The minimum
     * @param max The maximum
     * @return The clamped value
     */
    private static int clamp(final int val, final int min, final int max) {
        return Math.clamp(val, min, max);
    }
}
