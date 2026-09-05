package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.ansi.Theme;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Owns terminal lifecycle, invalidate polling, render, and global key hooks.
 */
public final class TuiApp implements AutoCloseable {
    public enum Signal { EXIT, IGNORE }

    private final TerminalRenderer terminal;
    private final ScreenManager host;
    private final Map<String, Consumer<UiHost>> globalKeys = new LinkedHashMap<>();
    private volatile boolean running;
    private int lastCols = -1;
    private int lastRows = -1;

    private TuiApp(final TerminalRenderer terminal) {
        this.terminal = terminal;
        this.host = new ScreenManager(terminal);
    }

    /**
     * Open a new TUI application instance.
     *
     * @return A new TuiApp instance with the terminal initialized
     * @throws IOException If the terminal cannot be initialized
     */
    public static TuiApp open() throws IOException {
        return new TuiApp(new TerminalRenderer());
    }

    /**
     * Register a global key handler that can exit the application.
     *
     * @param key The key to listen for
     * @param signal The signal to emit when the key is pressed
     * @return This TuiApp instance for method chaining
     */
    public TuiApp onGlobalKey(final String key, final Signal signal) {
        if (signal == Signal.EXIT) {
            globalKeys.put(key, ignored -> running = false);
        } else {
            globalKeys.put(key, ignored -> {});
        }
        return this;
    }

    /**
     * Register a global key handler.
     *
     * @param key The key to listen for
     * @param handler The handler to call when the key is pressed
     * @return This TuiApp instance for method chaining
     */
    public TuiApp onGlobalKey(final String key, final Consumer<UiHost> handler) {
        globalKeys.put(key, handler);
        return this;
    }

    /**
     * Set the color theme for the application.
     *
     * @param theme The theme to set
     * @return This TuiApp instance for method chaining
     */
    public TuiApp theme(final Theme theme) {
        Theme.setCurrent(theme);
        return this;
    }

    /**
     * Get the UI host for event delivery and toast management.
     *
     * @return The ScreenManager acting as the UI host
     */
    public UiHost host() {
        return host;
    }

    /**
     * Run the TUI application with the given initial screen.
     * Blocks until the application exits (via signal key or EOF).
     *
     * @param initialScreen The initial screen to display
     * @throws IOException If an I/O error occurs
     */
    public void run(final Screen initialScreen) throws IOException {
        host.push(initialScreen);
        rememberSize();
        renderFrame();
        running = true;
        while (running) {
            final UiMessage event = terminal.readEvent(() -> {
                final boolean postedOrResized = pump();
                if ((postedOrResized || host.isCurrentScreenInvalidated()) && host.peek() != null) {
                    renderFrame();
                }
            });
            if (event == null) {
                break;
            }

            if (event instanceof KeyPressed(final String key)) {
                final Screen current = host.peek();
                if (current == null || !current.capturesGlobalKey(key)) {
                    final Consumer<UiHost> hook = globalKeys.get(key);
                    if (hook != null) {
                        hook.accept(host);
                    }
                }
            }
            if (!running) {
                break;
            }

            host.forwardEvent(event);
            pump();
            final Screen current = host.peek();
            if (current == null) {
                break;
            }
            renderFrame();
        }
    }

    /**
     * Render the current frame with the screen, overlay, toasts, and status.
     */
    private void renderFrame() {
        terminal.render(
                host.peek(),
                host.peekOverlay(),
                host.visibleToasts(),
                host.peekStatus(),
                host.pendingToastCount()
        );
    }

    /**
     * Process posted events and check for resize events.
     *
     * @return true if the frame should be redrawn
     */
    private boolean pump() {
        final boolean resized = emitResizeIfNeeded();
        return resized || host.drainPostedEvents() > 0 || host.sweepToast();
    }

    /**
     * Remember the current terminal size.
     */
    private void rememberSize() {
        lastCols = terminal.getWidth();
        lastRows = terminal.getHeight();
    }

    /**
     * Emit a TerminalResized event if the terminal dimensions changed.
     *
     * @return true if the terminal was resized
     */
    private boolean emitResizeIfNeeded() {
        final int cols = terminal.getWidth();
        final int rows = terminal.getHeight();
        if (cols == lastCols && rows == lastRows) {
            return false;
        }
        lastCols = cols;
        lastRows = rows;
        if (host.peek() != null) {
            host.forwardEvent(new TerminalResized(cols, rows));
        }
        return true;
    }

    @Override
    public void close() throws IOException {
        terminal.close();
    }
}
