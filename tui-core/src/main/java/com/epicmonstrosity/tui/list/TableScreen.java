package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.ansi.Theme;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Stay-open or picker table over a {@link Collection}. Columns may be sortable
 * ({@code s} cycles them). Optional enqueue keys mark rows already queued.
 */
public class TableScreen<T> extends ScreenTemplate {
    private final TableModel<T> table;
    private Supplier<String> title;
    private Supplier<String> badge = () -> "";
    private Supplier<String> status = this::defaultStatus;
    private Consumer<T> onSelect;
    private Consumer<T> onEnqueue;
    private Predicate<T> enqueueAll;
    private Predicate<T> queued = item -> false;
    private String emptyMessage = "No items.";
    private boolean popOnSelect;
    private UiHost host;
    private boolean invalidated = true;
    private int viewportHeight = 10;

    /**
     * Create a table screen with callbacks.
     *
     * @param title The screen title
     * @param columns The table columns
     * @param items The initial items
     * @param onSelect Callback for selection
     */
    public TableScreen(final String title,
                       final List<TableColumn<T>> columns,
                       final Collection<? extends T> items,
                       final Consumer<T> onSelect) {
        this(title, columns, items);
        this.onSelect = onSelect == null ? ignored -> {} : onSelect;
        this.popOnSelect = true;
    }

    /**
     * Create a table screen.
     *
     * @param title The screen title
     * @param columns The table columns
     * @param items The initial items
     */
    public TableScreen(final String title,
                       final List<TableColumn<T>> columns,
                       final Collection<? extends T> items) {
        final String initial = title == null ? "Table" : title;
        this.title = () -> initial;
        this.table = new TableModel<>(columns, items);
        this.onSelect = ignored -> {};
        this.popOnSelect = false;
    }

    /**
     * Get the table model.
     *
     * @return The table model
     */
    public TableModel<T> model() {
        return table;
    }

    /**
     * Set the title supplier.
     *
     * @param title The title supplier
     * @return This TableScreen for chaining
     */
    public TableScreen<T> title(final Supplier<String> title) {
        if (title != null) {
            this.title = title;
        }
        return this;
    }

    /**
     * Set the badge supplier.
     *
     * @param badge The badge supplier
     * @return This TableScreen for chaining
     */
    public TableScreen<T> badge(final Supplier<String> badge) {
        this.badge = badge == null ? () -> "" : badge;
        return this;
    }

    /**
     * Set the status supplier.
     *
     * @param status The status supplier
     * @return This TableScreen for chaining
     */
    public TableScreen<T> status(final Supplier<String> status) {
        this.status = status == null ? this::defaultStatus : status;
        return this;
    }

    /**
     * Set the select callback.
     *
     * @param onSelect The select callback
     * @return This TableScreen for chaining
     */
    public TableScreen<T> onSelect(final Consumer<T> onSelect) {
        this.onSelect = onSelect == null ? ignored -> {} : onSelect;
        return this;
    }

    /**
     * Set to pop on select (default).
     *
     * @return This TableScreen for chaining
     */
    public TableScreen<T> popOnSelect() {
        this.popOnSelect = true;
        return this;
    }

    /**
     * Set whether to pop on select.
     *
     * @param popOnSelect Whether to pop
     * @return This TableScreen for chaining
     */
    public TableScreen<T> popOnSelect(final boolean popOnSelect) {
        this.popOnSelect = popOnSelect;
        return this;
    }

    /**
     * Set the enqueue callback.
     *
     * @param onEnqueue The enqueue callback
     * @return This TableScreen for chaining
     */
    public TableScreen<T> onEnqueue(final Consumer<T> onEnqueue) {
        this.onEnqueue = onEnqueue;
        return this;
    }

    /**
     * Set the enqueue-all predicate.
     *
     * @param enqueueAll The predicate for enqueue-all
     * @return This TableScreen for chaining
     */
    public TableScreen<T> enqueueMatching(final Predicate<T> enqueueAll) {
        this.enqueueAll = enqueueAll;
        return this;
    }

    /**
     * Set the queued predicate.
     *
     * @param queued The queued predicate
     * @return This TableScreen for chaining
     */
    public TableScreen<T> queued(final Predicate<T> queued) {
        this.queued = queued == null ? item -> false : queued;
        return this;
    }

    /**
     * Set the empty message.
     *
     * @param emptyMessage The empty message
     * @return This TableScreen for chaining
     */
    public TableScreen<T> emptyMessage(final String emptyMessage) {
        this.emptyMessage = emptyMessage == null || emptyMessage.isBlank() ? "No items." : emptyMessage;
        return this;
    }

    /**
     * Get the UI host.
     *
     * @return The UI host
     */
    protected UiHost host() {
        return host;
    }

    /**
     * Invalidate the screen.
     */
    protected void invalidate() {
        invalidated = true;
    }

    /**
     * Build the table layout.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        viewportHeight = Math.max(1, ctx.bodyRows() - 4);
        final Theme theme = Theme.current();
        final var body = new StringBuilder();
        body.append(theme.bold()).append(theme.info());
        body.append(headerRow());
        body.append(theme.reset()).append('\n');
        body.append("  ").repeat("─", Math.max(0, ctx.terminalCols() - 4)).append('\n');

        if (table.size() == 0) {
            body.append(theme.dim()).append("  ").append(emptyMessage).append('\n').append(theme.reset());
        } else {
            final List<T> window = table.visibleWindow(viewportHeight);
            final int from = table.scrollOffset();
            for (int i = 0; i < window.size(); i++) {
                paintRow(body, window.get(i), from + i, theme);
            }
            if (table.size() > viewportHeight) {
                body.append(theme.dim())
                        .append(scrollIndicator(table.scrollOffset(), viewportHeight, table.size()))
                        .append(theme.reset())
                        .append('\n');
            }
        }

        final String statusText = status.get();
        final String statusLine = statusText == null || statusText.isBlank() ? "" : "  " + statusText;
        final String badgeText = badge.get();
        return new ScreenLayout(
                headerBordered(title.get(), badgeText == null ? "" : badgeText, ctx.terminalCols() - 2),
                body.toString(),
                statusLine,
                footerError(host == null ? "" : host.errorBanner())
        );
    }

    /**
     * Get the key bindings for the table screen.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        final List<KeyBinding> bindings = new ArrayList<>();
        bindings.add(KeyBinding.primary("UP", "↑", "up", table::moveUp));
        bindings.add(KeyBinding.primary("DOWN", "↓", "down", table::moveDown));
        bindings.add(KeyBinding.primary("ENTER", "Enter", "select", this::accept));
        if (onEnqueue != null) {
            bindings.add(KeyBinding.primary("a", "a", "queue", this::enqueueSelected));
            bindings.add(KeyBinding.primary("p", "p", "enq all", this::enqueueMatchingRows));
        }
        if (table.sortable()) {
            bindings.add(KeyBinding.primary("s", "s", "sort", table::cycleSort));
        }
        bindings.add(KeyBinding.secondary("ESC", "Esc", "back", this::back));
        bindings.add(KeyBinding.secondary("HOME", "↖", "home", table::moveFirst));
        bindings.add(KeyBinding.secondary("END", "↘", "end", table::moveLast));
        bindings.add(KeyBinding.secondary("PAGE_UP", "⇑", "pg up", () -> table.moveCursor(-viewportHeight)));
        bindings.add(KeyBinding.secondary("PAGE_DOWN", "⇓", "pg down", () -> table.moveCursor(viewportHeight)));
        return bindings;
    }

    /**
     * Called when the table screen enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the table screen exits.
     */
    @Override
    public void onExit() {
    }

    /**
     * Handle UI events for the table screen.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof KeyPressed(String key)) {
            handleInput(key);
            invalidated = true;
        }
    }

    /**
     * Check if the table screen should be invalidated.
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
        table.selectEntry().ifPresent(onSelect);
        if (popOnSelect && host != null) {
            host.navigation().pop();
        }
    }

    /**
     * Enqueue the selected item.
     */
    private void enqueueSelected() {
        if (onEnqueue == null) {
            return;
        }
        table.selectEntry().ifPresent(onEnqueue);
    }

    /**
     * Enqueue all matching rows.
     */
    private void enqueueMatchingRows() {
        if (onEnqueue == null) {
            return;
        }
        for (T item : table.matching(enqueueAll)) {
            onEnqueue.accept(item);
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
     * Get the default status.
     *
     * @return The status
     */
    private String defaultStatus() {
        return table.statusMessage();
    }

    /**
     * Build the header row.
     *
     * @return The header row string
     */
    private String headerRow() {
        final StringBuilder row = new StringBuilder("  ");
        for (int i = 0; i < table.columns().size(); i++) {
            if (i > 0) {
                row.append("  ");
            }
            final TableColumn<T> column = table.columns().get(i);
            String label = column.header() == null ? "" : column.header();
            if (table.sorted() && i == table.sortColumn()) {
                label = label + (table.sortAscending() ? " ▲" : " ▼");
            }
            row.append(truncate(label, Math.max(1, column.width())));
        }
        return row.toString();
    }

    /**
     * Paint a row.
     *
     * @param body The StringBuilder to append to
     * @param item The item
     * @param index The row index
     * @param theme The color theme
     */
    private void paintRow(final StringBuilder body, final T item, final int index, final Theme theme) {
        final boolean selected = index == table.getCursor();
        if (selected) {
            body.append(theme.selection()).append(theme.bold());
        }
        final boolean inQueue = queued.test(item);
        final String mark = inQueue ? "» " : "  ";
        body.append(mark);
        for (int i = 0; i < table.columns().size(); i++) {
            if (i > 0) {
                body.append("  ");
            }
            final TableColumn<T> column = table.columns().get(i);
            final String raw = column.value() == null ? "" : String.valueOf(column.value().apply(item));
            final String cell = truncate(raw, Math.max(1, column.width()));
            final String style = column.style() == null ? "" : nz(column.style().apply(item));
            if (!style.isEmpty()) {
                body.append(style).append(cell).append(theme.reset());
                if (selected) {
                    body.append(theme.selection()).append(theme.bold());
                }
            } else {
                body.append(cell);
            }
        }
        if (selected) {
            body.append(theme.reset());
        }
        body.append('\n');
    }

    /**
     * Return value or empty string if null.
     *
     * @param value The value
     * @return The value or empty string
     */
    private static String nz(final String value) {
        return value == null ? "" : value;
    }
}
