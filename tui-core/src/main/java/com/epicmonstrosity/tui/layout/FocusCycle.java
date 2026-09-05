package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.UiMessage;

import java.util.List;

/**
 * Named focus regions on a single screen. Tab / Shift+Tab cycle;
 * other keys stay with the app so the active region can handle them.
 */
public final class FocusCycle {
    private final List<String> regions;
    private int index;

    /**
     * Create a focus cycle.
     *
     * @param names The focus region names
     * @throws IllegalArgumentException If no names are provided
     */
    public FocusCycle(final String... names) {
        if (names == null || names.length == 0) {
            throw new IllegalArgumentException("at least one focus region is required");
        }
        this.regions = List.of(names);
    }

    /**
     * Get the current focus region.
     *
     * @return The current region name
     */
    public String current() {
        return regions.get(index);
    }

    /**
     * Check if the current region matches the given name.
     *
     * @param name The name to check
     * @return true if current
     */
    public boolean is(final String name) {
        return current().equals(name);
    }

    /**
     * Get the current index.
     *
     * @return The index
     */
    public int index() {
        return index;
    }

    /**
     * Get the number of regions.
     *
     * @return The size
     */
    public int size() {
        return regions.size();
    }

    /**
     * Get all region names.
     *
     * @return The list of names
     */
    public List<String> names() {
        return regions;
    }

    /**
     * Move to the next region.
     */
    public void next() {
        index = (index + 1) % regions.size();
    }

    /**
     * Move to the previous region.
     */
    public void previous() {
        index = (index - 1 + regions.size()) % regions.size();
    }

    /**
     * Focus a specific region by name.
     *
     * @param name The name to focus
     */
    public void focus(final String name) {
        final int found = regions.indexOf(name);
        if (found >= 0) {
            index = found;
        }
    }

    /**
     * Handle a UI message, consuming Tab/Shift+Tab if applicable.
     *
     * @param msg The message to handle
     * @return true if Tab/Shift+Tab was consumed
     */
    public boolean handle(final UiMessage msg) {
        if (msg instanceof KeyPressed(String key)) {
            if ("TAB".equals(key)) {
                next();
                return true;
            }
            if ("SHIFT_TAB".equals(key)) {
                previous();
                return true;
            }
        }
        return false;
    }
}
