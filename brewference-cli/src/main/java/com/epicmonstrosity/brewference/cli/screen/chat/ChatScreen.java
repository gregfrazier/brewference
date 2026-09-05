package com.epicmonstrosity.brewference.cli.screen.chat;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.TerminalResized;
import com.epicmonstrosity.tui.Tick;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import com.epicmonstrosity.tui.ansi.Theme;
import com.epicmonstrosity.tui.overlay.HelpOverlay;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;

import java.util.List;
import java.util.function.Consumer;

/**
 * A chat screen that renders messages and handles input.
 */
public final class ChatScreen extends ScreenTemplate {
    private final String title;
    private final ChatModel model;
    private final Consumer<String> onSubmit;
    private final ChatCommands commands;
    private final Runnable onExit;
    private Runnable stopHandler = () -> {};
    private UiHost host;
    private boolean invalidated = true;
    private int lastRenderedMessages = -1;
    private int viewportHeight = 10;
    private int cols = 80;

    /**
     * Create a chat screen with standard commands.
     *
     * @param title The screen title
     * @param model The chat model
     * @param onSubmit Callback for user submission
     */
    public ChatScreen(final String title, final ChatModel model, final Consumer<String> onSubmit, final Runnable onExit) {
        this(title, model, onSubmit, ChatCommands.standard(), onExit);
    }

    /**
     * Create a chat screen with custom commands.
     *
     * @param title The screen title
     * @param model The chat model
     * @param onSubmit Callback for user submission
     * @param commands The command registry
     */
    public ChatScreen(final String title,
                      final ChatModel model,
                      final Consumer<String> onSubmit,
                      final ChatCommands commands,
                      final Runnable onExit) {
        this.title = title == null ? "Chat" : title;
        this.model = model;
        this.onSubmit = onSubmit == null ? text -> {} : onSubmit;
        this.commands = commands == null ? ChatCommands.standard() : commands;
        this.onExit = onExit == null ? () -> {} : onExit;
    }

    /**
     * Get the chat model.
     *
     * @return The chat model
     */
    public ChatModel model() {
        return model;
    }

    /**
     * Get the command registry.
     *
     * @return The command registry
     */
    public ChatCommands commands() {
        return commands;
    }

    /**
     * Register a custom command.
     *
     * @param command The command to register
     * @return This ChatScreen for chaining
     */
    public ChatScreen command(final ChatCommand command) {
        commands.register(command);
        return this;
    }

    /**
     * Register a handler for stopping an in-progress reply.
     *
     * @param handler The stop handler
     * @return This ChatScreen for chaining
     */
    public ChatScreen stopHandler(final Runnable handler) {
        this.stopHandler = handler == null ? () -> {} : handler;
        return this;
    }

    /**
     * Check if the key should be captured (printable chars only).
     *
     * @param key The key to check
     * @return true if printable (captured)
     */
    @Override
    public boolean capturesGlobalKey(final String key) {
        return isPrintable(key);
    }

    /**
     * Build the chat layout with messages and input.
     *
     * @param ctx The view context
     * @return The screen layout
     */
    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        cols = Math.max(1, ctx.terminalCols() - 2);
        viewportHeight = Math.max(1, ctx.bodyRows() - 4);
        lastRenderedMessages = model.messages().size();
        final var body = new StringBuilder();
        for (final ChatLine line : model.visibleChatLines(cols, viewportHeight)) {
            body.append(colorLine(line)).append('\n');
        }
        final int total = model.totalWrapped(cols);
        if (total > viewportHeight) {
            body.append(DIM)
                    .append(scrollIndicator(model.scrollOffset(), viewportHeight, total))
                    .append(RESET)
                    .append('\n');
        }
        body.append('\n')
                .append(BOLD)
                .append("> ")
                .append(RESET)
                .append(model.inputBuffer().replace("\n", "⏎"))
                .append(BG_SEL)
                .append(' ')
                .append(RESET);

        return new ScreenLayout(
                headerBordered(title, "", ctx.terminalCols() - 2),
                body.toString(),
                "",
                footerError(host == null ? "" : host.errorBanner())
        );
    }

    /**
     * Get the key bindings for the chat screen.
     *
     * @return List of key bindings
     */
    @Override
    public List<KeyBinding> keyBindings() {
        final var bindings = new java.util.ArrayList<KeyBinding>();
        bindings.add(KeyBinding.primary("ENTER", "Enter", "send", this::submit));
        bindings.add(KeyBinding.primary("CTRL_J", "Ctrl+J", "newline", model::newline));
        bindings.add(KeyBinding.primary("UP", "↑", "scroll up", () -> model.scrollBy(-1, cols, viewportHeight)));
        bindings.add(KeyBinding.primary("DOWN", "↓", "scroll down", () -> model.scrollBy(1, cols, viewportHeight)));
        bindings.add(KeyBinding.primary("FN_2", "F2", "stop", stopHandler));
        bindings.add(KeyBinding.secondary("PAGE_UP", "⇑", "pg up", () -> model.pageUp(viewportHeight)));
        bindings.add(KeyBinding.secondary("PAGE_DOWN", "⇓", "pg down", () -> model.pageDown(viewportHeight)));
        bindings.add(KeyBinding.secondary("FN_1", "F1", "help", this::openHelp));
        //if (model.streaming()) {

        //}
        bindings.add(KeyBinding.secondary("ESC", "Esc", "back", this::back));
        return List.copyOf(bindings);
    }

    /**
     * Called when the chat screen enters.
     *
     * @param host The UI host
     */
    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    /**
     * Called when the chat screen exits.
     */
    @Override
    public void onExit() {
        onExit.run();
    }

    /**
     * Handle UI events (paste, key, tick, resize).
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
        if (msg instanceof Tick || msg instanceof TerminalResized) {
            invalidated = true;
            return;
        }
        if (msg instanceof KeyPressed(final String key)) {
            if (isPrintable(key)) {
                model.type(key);
            } else if ("BACKSPACE".equals(key)) {
                model.backspace();
            } else {
                handleInput(key);
            }
            invalidated = true;
        }
    }

    /**
     * Check if the screen should be invalidated.
     *
     * @return true if invalidated
     */
    @Override
    public boolean isInvalidated() {
        return invalidated || model.streaming() || model.messages().size() != lastRenderedMessages;
    }

    /**
     * Submit the input as a chat message.
     */
    private void submit() {
        final String text = model.takeInput().trim();
        if (text.isEmpty()) {
            return;
        }
        if (model.streaming()) {
            onSubmit.accept(text);
            return;
        }
        final ChatCommandContext ctx = new ChatCommandContext(text, "", "", model, host, commands);
        if (commands.dispatch(text, ctx)) {
            return;
        }
        model.submitUser(text);
        onSubmit.accept(text);
    }

    /**
     * Open the help overlay.
     */
    private void openHelp() {
        if (host == null || host.peekOverlay() != null) {
            return;
        }
        host.pushOverlay(HelpOverlay.from(this));
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
        return key != null && key.length() == 1 && key.charAt(0) >= 32;
    }

    /**
     * Color a chat line according to its role.
     *
     * @param line The chat line to color
     * @return The colored line string
     */
    private String colorLine(final ChatLine line) {
        final Theme theme = Theme.current();
        final String style = switch (line.role() == null ? "" : line.role()) {
            case ChatMessage.ASSISTANT -> theme.info();
            case ChatMessage.SYSTEM -> theme.warning();
            case ChatMessage.TOOL -> theme.muted();
            default -> theme.accent();
        };
        final String gutter = theme.bold() + theme.paint(style, line.gutter());
        final String sep = theme.paint(theme.muted(), ChatLine.SEPARATOR);
        final String text = model.generating() && !line.continuation() && ChatMessage.ASSISTANT.equals(line.role())
                ? theme.paint(theme.muted(), line.text())
                : line.text();
        return gutter + sep + text;
    }
}
