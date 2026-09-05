package com.epicmonstrosity.tui.layout;

/**
 * Named first/second panes for a body region. Query {@link #firstWidth()}
 * (and friends) before rendering each block so lists wrap to the pane,
 * then {@link #render()} stitches them.
 */
public final class BodySlots {
    /** Vertical axis (top/bottom split). */
    public enum Axis { VERTICAL, HORIZONTAL }

    /** The axis direction. */
    private final Axis axis;
    /** The terminal columns or rows. */
    private final int cols;
    /** The terminal rows or columns. */
    private final int rows;
    /** The first pane percentage. */
    private final int firstPercent;
    /** The first pane content. */
    private String first = "";
    /** The second pane content. */
    private String second = "";

    /**
     * Construct a BodySlots instance.
     *
     * @param axis The axis direction
     * @param cols The columns (for VERTICAL) or rows (for HORIZONTAL)
     * @param rows The rows (for VERTICAL) or columns (for HORIZONTAL)
     * @param firstPercent The first pane percentage
     */
    private BodySlots(final Axis axis, final int cols, final int rows, final int firstPercent) {
        this.axis = axis;
        this.cols = Math.max(0, cols);
        this.rows = Math.max(0, rows);
        this.firstPercent = firstPercent;
    }

    /**
     * Create a vertical BodySlots with default percent.
     *
     * @param cols The columns
     * @param rows The rows
     * @return The BodySlots instance
     */
    public static BodySlots vertical(final int cols, final int rows) {
        return vertical(cols, rows, Split.DEFAULT_PERCENT);
    }

    /**
     * Create a vertical BodySlots.
     *
     * @param cols The columns
     * @param rows The rows
     * @param firstPercent The first pane percentage
     * @return The BodySlots instance
     */
    public static BodySlots vertical(final int cols, final int rows, final int firstPercent) {
        return new BodySlots(Axis.VERTICAL, cols, rows, firstPercent);
    }

    /**
     * Create a horizontal BodySlots with default percent.
     *
     * @param cols The columns
     * @param rows The rows
     * @return The BodySlots instance
     */
    public static BodySlots horizontal(final int cols, final int rows) {
        return horizontal(cols, rows, Split.DEFAULT_PERCENT);
    }

    /**
     * Create a horizontal BodySlots.
     *
     * @param cols The columns
     * @param rows The rows
     * @param firstPercent The first pane percentage
     * @return The BodySlots instance
     */
    public static BodySlots horizontal(final int cols, final int rows, final int firstPercent) {
        return new BodySlots(Axis.HORIZONTAL, cols, rows, firstPercent);
    }

    /**
     * Get the axis direction.
     *
     * @return The axis
     */
    public Axis axis() {
        return axis;
    }

    /**
     * Get the first pane percentage.
     *
     * @return The percentage
     */
    public int firstPercent() {
        return firstPercent;
    }

    /**
     * Get the split sizes.
     *
     * @return The SplitSizes
     */
    public SplitSizes sizes() {
        return axis == Axis.VERTICAL ? Split.sizes(cols, firstPercent) : Split.sizes(rows, firstPercent);
    }

    /**
     * Get the first pane width (for vertical) or height (for horizontal).
     *
     * @return The width
     */
    public int firstWidth() {
        return axis == Axis.VERTICAL ? sizes().first() : cols;
    }

    /**
     * Get the first pane height (for vertical) or width (for horizontal).
     *
     * @return The height
     */
    public int firstHeight() {
        return axis == Axis.VERTICAL ? rows : sizes().first();
    }

    /**
     * Get the second pane width (for vertical) or height (for horizontal).
     *
     * @return The width
     */
    public int secondWidth() {
        return axis == Axis.VERTICAL ? sizes().second() : cols;
    }

    /**
     * Get the second pane height (for vertical) or width (for horizontal).
     *
     * @return The height
     */
    public int secondHeight() {
        return axis == Axis.VERTICAL ? rows : sizes().second();
    }

    /**
     * Set the left/top pane content.
     *
     * @param content The content
     * @return This BodySlots for chaining
     */
    public BodySlots left(final String content) {
        this.first = content == null ? "" : content;
        return this;
    }

    /**
     * Set the right/bottom pane content.
     *
     * @param content The content
     * @return This BodySlots for chaining
     */
    public BodySlots right(final String content) {
        this.second = content == null ? "" : content;
        return this;
    }

    /**
     * Set the top pane content (alias for left).
     *
     * @param content The content
     * @return This BodySlots for chaining
     */
    public BodySlots top(final String content) {
        return left(content);
    }

    /**
     * Set the bottom pane content (alias for right).
     *
     * @param content The content
     * @return This BodySlots for chaining
     */
    public BodySlots bottom(final String content) {
        return right(content);
    }

    /**
     * Set a pane by name.
     *
     * @param name The pane name (left, right, top, bottom, first, second)
     * @param content The content
     * @return This BodySlots for chaining
     * @throws IllegalArgumentException If the name is invalid
     */
    public BodySlots put(final String name, final String content) {
        if (name == null) {
            throw new IllegalArgumentException("unknown slot: null");
        }
        return switch (name) {
            case "left", "top", "first" -> left(content);
            case "right", "bottom", "second" -> right(content);
            default -> throw new IllegalArgumentException("unknown slot: " + name);
        };
    }

    /**
     * Render the two panes.
     *
     * @return The rendered string
     */
    public String render() {
        if (axis == Axis.VERTICAL) {
            return Split.vertical(first, second, cols, rows, firstPercent);
        }
        return Split.horizontal(first, second, cols, rows, firstPercent);
    }
}
