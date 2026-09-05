package com.epicmonstrosity.tui.form;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.TerminalResized;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.list.FilePickerScreen;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

/**
 * A form screen for editing POJOs and records.
 * @param <T> The type of the form data
 */
public final class FormScreen<T> extends ScreenTemplate {
    private final String title;
    private final FormModel<T> model;
    private final Consumer<T> onSubmit;
    private UiHost host;
    private boolean invalidated = true;
    private boolean popOnSubmit;

    /**
     * Create a form screen.
     *
     * @param title The screen title
     * @param type The class type
     * @param instance The original instance (can be null)
     * @param onSubmit Callback for successful submission
     */
    public FormScreen(final String title, final Class<T> type, final T instance, final Consumer<T> onSubmit) {
        this.title = title == null ? "Form" : title;
        this.model = new FormModel<>(type, instance);
        this.onSubmit = onSubmit == null ? value -> {} : onSubmit;
    }

    /**
     * Add a custom constraint to the form.
     *
     * @param constraint The constraint to add
     * @return This FormScreen for chaining
     */
    public FormScreen<T> constrain(final FormConstraint constraint) {
        model.constrain(constraint);
        return this;
    }

    /**
     * Set to pop the screen on successful submission.
     *
     * @return This FormScreen for chaining
     */
    public FormScreen<T> popOnSubmit() {
        this.popOnSubmit = true;
        return this;
    }

    /**
     * Build the form layout with fields.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        final var body = new StringBuilder();
        final List<FormFieldState> fields = model.fields();
        for (int i = 0; i < fields.size(); i++) {
            final FormFieldState field = fields.get(i);
            final String suffix = field.required() ? " *" : "";
            final String value = field.displayValue() + (field.readOnly() ? " (read only)" : "");
            configField(body, model.cursor(), i, field.label() + suffix, value);
            if (field.error() != null && !field.error().isBlank()) {
                body.append(RED).append("      ").append(field.error()).append(RESET).append('\n');
            } else if (field.help() != null && !field.help().isBlank() && i == model.cursor()) {
                body.append(DIM).append("      ").append(field.help()).append(RESET).append('\n');
            }
        }
        final String status = model.status().isBlank() ? "" : "  " + DIM + model.status() + RESET;
        return new ScreenLayout(
                headerBordered(title, "", ctx.terminalCols() - 2),
                body.toString(),
                status,
                footerError(host == null ? "" : host.errorBanner())
        );
    }

    /**
     * Get the key bindings for the form screen.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        return List.of(
                KeyBinding.primary("UP", "↑", "up", () -> model.moveBy(-1)),
                KeyBinding.primary("DOWN", "↓", "down", () -> model.moveBy(1)),
                KeyBinding.primary("SPACE", "Space", "toggle", this::activateOrPick),
                KeyBinding.primary("CTRL_S", "Ctrl+S", "save", this::submit),
                KeyBinding.secondary("ENTER", "Enter", "newline / pick", this::enterOnField),
                KeyBinding.secondary("ESC", "Esc", "back", this::back)
        );
    }

    /**
     * Called when the form screen enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the form screen exits.
     */
    @Override
    public void onExit() {
    }

    /**
     * Handle UI events for the form.
     *
     * @param msg The event to handle
     */
    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof Pasted(final String text)) {
            model.paste(text);
            invalidated = true;
            return;
        }
        if (msg instanceof TerminalResized) {
            invalidated = true;
            return;
        }
        if (msg instanceof KeyPressed(final String key)) {
            if (isPrintable(key)) {
                model.type(key);
            } else if ("BACKSPACE".equals(key)) {
                model.backspace();
            } else if ("CTRL_J".equals(key)) {
                model.newline();
            } else if ("ENTER".equals(key) || " ".equals(key) || "SPACE".equals(key)) {
                enterOnField();
            } else {
                handleInput(key);
            }
            invalidated = true;
        }
    }

    /**
     * Check if the form should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    /**
     * Enter on the current field (newline for multiline, activate otherwise).
     */
    private void enterOnField() {
        final FormFieldState field = model.selected();
        if (field != null && field.multiline()) {
            model.newline();
            return;
        }
        activateOrPick();
    }

    /**
     * Activate or pick a file for Path fields.
     */
    private void activateOrPick() {
        final FormFieldState field = model.selected();
        if (field != null && field.type() == Path.class && host != null && !field.readOnly()) {
            final Path start = startPath(field);
            host.navigation().push(FilePickerScreen.any("Pick " + field.label(), start, path -> {
                field.setValue(path);
                invalidated = true;
            }));
            return;
        }
        model.activate();
    }

    /**
     * Get the starting path for a file picker.
     *
     * @param field The field
     * @return The starting path
     */
    private static Path startPath(final FormFieldState field) {
        if (field.value() instanceof final Path path) {
            return path;
        }
        final String text = field.text();
        if (text != null && !text.isBlank()) {
            return Path.of(text.trim());
        }
        return Path.of(System.getProperty("user.dir"));
    }

    /**
     * Submit the form.
     */
    private void submit() {
        final T written = model.submit();
        if (written != null) {
            onSubmit.accept(written);
            if (popOnSubmit && host != null) {
                host.navigation().pop();
            }
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
     * Check if a key is printable.
     *
     * @param key The key to check
     * @return true if printable
     */
    private static boolean isPrintable(final String key) {
        return key.length() == 1 && key.charAt(0) >= 32 && !" ".equals(key);
    }
}
