package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public final class FilePickerScreen extends ScreenTemplate {
    private final String title;
    private final FilePickerModel model;
    private final Consumer<Path> onPick;
    private UiHost host;
    private boolean invalidated = true;
    private int viewportHeight = 10;

    /**
     * Create a file picker screen.
     *
     * @param title The screen title
     * @param model The file picker model
     * @param onPick Callback for picked path
     */
    private FilePickerScreen(final String title, final FilePickerModel model, final Consumer<Path> onPick) {
        this.title = title == null ? "Pick a path" : title;
        this.model = model;
        this.onPick = onPick == null ? ignored -> {} : onPick;
    }

    /**
     * Create a file picker for files.
     *
     * @param title The screen title
     * @param start The starting directory
     * @param onPick Callback for picked path
     * @return The FilePickerScreen
     */
    public static FilePickerScreen files(final String title, final Path start, final Consumer<Path> onPick) {
        return new FilePickerScreen(title, FilePickerModel.files(start), onPick);
    }

    /**
     * Create a file picker for directories.
     *
     * @param title The screen title
     * @param start The starting directory
     * @param onPick Callback for picked path
     * @return The FilePickerScreen
     */
    public static FilePickerScreen directories(final String title, final Path start, final Consumer<Path> onPick) {
        return new FilePickerScreen(title, FilePickerModel.directories(start), onPick);
    }

    /**
     * Create a file picker for files and directories.
     *
     * @param title The screen title
     * @param start The starting directory
     * @param onPick Callback for picked path
     * @return The FilePickerScreen
     */
    public static FilePickerScreen any(final String title, final Path start, final Consumer<Path> onPick) {
        return new FilePickerScreen(title, FilePickerModel.any(start), onPick);
    }

    /**
     * Build the file picker layout.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        viewportHeight = Math.max(1, ctx.bodyRows() - 3);
        final var body = new StringBuilder();
        body.append(DIM).append("  ").append(model.current()).append(RESET).append('\n');
        body.append(DIM).append("  Filter: ").append(RESET).append(model.filter()).append('\n');
        final List<FilePickerModel.Entry> window = model.visibleWindow(viewportHeight);
        final int from = model.scrollOffset();
        for (int i = 0; i < window.size(); i++) {
            final FilePickerModel.Entry entry = window.get(i);
            configField(body, model.getCursor(), from + i, marker(entry), entry.label());
        }
        if (model.getEntriesSize() > viewportHeight) {
            body.append(DIM)
                    .append(scrollIndicator(model.scrollOffset(), viewportHeight, model.getEntriesSize()))
                    .append(RESET)
                    .append('\n');
        }
        final String error = model.error().isBlank()
                ? (host == null ? "" : host.errorBanner())
                : model.error();
        return new ScreenLayout(
                headerBordered(title, "", ctx.terminalCols() - 2),
                body.toString(),
                "",
                footerError(error)
        );
    }

    /**
     * Get the key bindings for the file picker.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        final List<KeyBinding> bindings = new ArrayList<>();
        bindings.add(KeyBinding.primary("UP", "↑", "up", model::moveUp));
        bindings.add(KeyBinding.primary("DOWN", "↓", "down", model::moveDown));
        bindings.add(KeyBinding.primary("ENTER", "Enter", "open / pick", this::enterOrPick));
        if (model.mode() != FilePickerModel.Mode.FILES) {
            bindings.add(KeyBinding.primary("CTRL_S", "Ctrl+S", "this folder", this::pickCurrent));
        }
        bindings.add(KeyBinding.secondary("PAGE_UP", "⇑", "pg up", () -> model.moveCursor(-viewportHeight)));
        bindings.add(KeyBinding.secondary("PAGE_DOWN", "⇓", "pg down", () -> model.moveCursor(viewportHeight)));
        bindings.add(KeyBinding.secondary("HOME", "↖", "home", model::moveFirst));
        bindings.add(KeyBinding.secondary("END", "↘", "end", model::moveLast));
        bindings.add(KeyBinding.secondary("ESC", "Esc", "back", this::back));
        return bindings;
    }

    /**
     * Called when the file picker enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the file picker exits.
     */
    @Override
    public void onExit() {
    }

    /**
     * Handle UI events for the file picker.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof Pasted(String text)) {
            model.typeFilter(text.replace("\n", ""));
            invalidated = true;
            return;
        }
        if (msg instanceof KeyPressed(String key)) {
            if (isPrintable(key)) {
                model.typeFilter(key);
            } else if ("BACKSPACE".equals(key)) {
                model.backspaceFilter();
            } else if ("ESC".equals(key) && !model.filter().isEmpty()) {
                model.setFilter("");
            } else {
                handleInput(key);
            }
            invalidated = true;
        }
    }

    /**
     * Check if the file picker should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Enter or pick a path.
     */
    private void enterOrPick() {
        if (model.enterSelected()) {
            return;
        }
        model.pickSelected().ifPresent(this::choose);
    }

    /**
     * Pick the current directory.
     */
    private void pickCurrent() {
        model.pickCurrentDirectory().ifPresent(this::choose);
    }

    /**
     * Choose a path.
     *
     * @param path The chosen path
     */
    private void choose(final Path path) {
        onPick.accept(path);
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
     * Get the marker for an entry.
     *
     * @param entry The entry
     * @return The marker
     */
    private String marker(final FilePickerModel.Entry entry) {
        return switch (entry.kind()) {
            case PARENT -> "↑";
            case DIRECTORY -> "▸";
            case FILE -> " ";
        };
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
