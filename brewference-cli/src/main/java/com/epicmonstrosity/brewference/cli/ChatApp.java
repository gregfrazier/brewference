package com.epicmonstrosity.brewference.cli;

import com.epicmonstrosity.brewference.cli.screen.chat.*;
import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.generation.GenerationResult;
import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.runtime.ModelSession;
import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.chat.ChatConversation;
import com.epicmonstrosity.tui.UiHost;

import java.io.IOException;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Wires brewference's generation stack to the chat screen.
 */
public final class ChatApp implements TokenConsumer {
    private static final String BUSY_TOAST = "Model is busy - wait for the current reply";

    private final PromptTemplate template;
    private final GenerationOptions options;
    private final String systemPrompt;
    private final ChatModel model = new ChatModel();
    private final AtomicBoolean busy = new AtomicBoolean(false);

    private ModelSession session;
    private String filename;
    private ChatConversation conversation;
    private UiHost host;
    private ExecutorService worker;
    private volatile boolean appOpen;
    private boolean systemPrompted;

    // Only touched on the worker thread during a generate call.
    private final StringBuilder currentReply = new StringBuilder();
    private boolean echoOpen;

    public ChatApp(final PromptTemplate template,
                   final GenerationOptions options,
                   final String systemPrompt) {
        this.template = template;
        this.options = options;
        this.systemPrompt = systemPrompt;
        this.conversation = newConversation();
    }

    /**
     * Attach the loaded model session. Must be called before run().
     */
    public void setSession(final ModelSession session, final String filename) {
        this.session = session;
        this.filename = filename;
    }

    /**
     * Open the TUI chat screen and block until it exits.
     */
    public void run(final UiHost host, final Runnable onExit) throws IOException {
        if (session == null) {
            throw new IllegalStateException("setSession(...) must be called before run()");
        }

        final ChatScreen screen = new ChatScreen(
                "Brewference chat - " + filename, model, this::onSubmit, ChatCommands.standard(),
                () -> {
                    appOpen = false;
                    if (worker != null) {
                        worker.shutdownNow();
                    }
                    onExit.run();
                });
        screen.stopHandler(() -> session.requestStop());
        screen.command(new ExitCommand());
        screen.command(new ClearCommand());
        screen.command(new DetailsCommand());
        screen.command(new MetadataCommand());
        screen.command(new TensorsCommand());

        worker = Executors.newSingleThreadExecutor();
        this.host = host;
        appOpen = true;
        host.navigation().push(screen);
    }

    /**
     * UI-thread callback for submitted input. Adds the message to the conversation,
     * renders the prompt (first turn: full conversation; later turns: latest message only),
     * and submits generation to the worker thread.
     */
    private void onSubmit(final String text) {
        if (!busy.compareAndSet(false, true)) {
            host.showToast(BUSY_TOAST);
            return;
        }

        final com.epicmonstrosity.brewference.template.chat.ChatMessage chatMessage =
                conversation.addUserMessage(text);
        final String prompt = template.renderForCompletion(
                systemPrompted ? Collections.singletonList(chatMessage) : conversation.messages());
        systemPrompted = true;

        try {
            worker.submit(() -> {
                currentReply.setLength(0);
                echoOpen = false;
                try {
                    if (!options.isEchoPrompt()) {
                        model.startGenerating();
                    }
                    session.generate(prompt, options);
                } catch (final Exception e) {
                    model.finishStreaming();
                    model.append(ChatMessage.system("Error: " + e));
                } finally {
                    busy.set(false);
                }
            });
        } catch (final RejectedExecutionException e) {
            busy.set(false);
            throw e;
        }
    }

    private ChatConversation newConversation() {
        final ChatConversation fresh = new ChatConversation();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            fresh.addSystemMessage(systemPrompt);
        }
        return fresh;
    }

    @Override
    public void onDebug(final String debug) {
        if (!appOpen) {
            System.out.println(debug);
        }
    }

    @Override
    public void onPrefillTotal(final int total) {
        if (options.isEchoPrompt()) {
            echoOpen = true;
            model.startStreaming(ChatMessage.TOOL);
        }
    }

    @Override
    public void onPrefillToken(final int position, final int tokenId, final String tokenText) {
        if (echoOpen) {
            model.appendDelta("tokenId: " + tokenId + " tokenText: " + tokenText + "\n");
        }
    }

    @Override
    public void onGeneratedToken(final int position, final int tokenId, final String tokenText) {
        if (echoOpen) {
            echoOpen = false;
            model.finishStreaming();
        }
        currentReply.append(tokenText);
        model.appendDelta(tokenText);
    }

    @Override
    public void onComplete(final GenerationResult result) {
        model.finishStreaming();
        conversation.addAssistantMessage(currentReply.toString());
        if (result.isStoppedByUser()) {
            model.append(ChatMessage.system(
                    "[stopped by user after " + result.getGeneratedTokenCount() + " tokens]"));
        } else {
            model.append(ChatMessage.system(
                    "[generated: " + result.getGeneratedTokenCount()
                            + " tokens, eos: " + result.isStoppedByEos()
                            + ", elapsedMs: " + (result.getElapsedNanos() / 1_000_000.0) + "]"));
        }
    }

    /** /exit — pop the only screen so TuiApp.run returns. */
    private static final class ExitCommand implements ChatCommand {
        @Override
        public String name() {
            return "exit";
        }

        @Override
        public String description() {
            return "Quit the chat";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            ctx.host().navigation().pop();
        }
    }

    /** /clear — reset conversation, session and display (blocked while busy). */
    private final class ClearCommand implements ChatCommand {
        @Override
        public String name() {
            return "clear";
        }

        @Override
        public String description() {
            return "Reset the conversation";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            if (busy.get()) {
                ctx.host().showToast(BUSY_TOAST);
                return;
            }
            synchronized (ChatApp.this) {
                conversation = newConversation();
                session.clear();
                model.clear();
                systemPrompted = false;
            }
            model.append(ChatMessage.system("(context cleared)"));
        }
    }

    /** /details — append the model config as a system message. */
    private final class DetailsCommand implements ChatCommand {
        @Override
        public String name() {
            return "details";
        }

        @Override
        public String description() {
            return "Print model details";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            ctx.model().append(ChatMessage.system("Model details:\n" + session.getConfig()));
        }
    }

    /** /metadata [key] — list metadata keys, or show one value. */
    private final class MetadataCommand implements ChatCommand {
        @Override
        public String name() {
            return "metadata";
        }

        @Override
        public String description() {
            return "Print model metadata";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            final String key = ctx.args();
            if (key.isEmpty()) {
                final StringBuilder sb = new StringBuilder("Model metadata:\n");
                session.getConfig().getMetadata().keySet().stream()
                        .sorted()
                        .forEach(k -> sb.append(k).append('\n'));
                ctx.model().append(ChatMessage.system(sb.toString()));
            } else {
                final Object value = session.getConfig().getMetadata().get(key);
                ctx.model().append(ChatMessage.system(
                        "Model metadata " + key + "\n" + (value != null ? value : "not found")));
            }
        }
    }

    /** /tensors — append the tensor name list as a system message. */
    private final class TensorsCommand implements ChatCommand {
        @Override
        public String name() {
            return "tensors";
        }

        @Override
        public String description() {
            return "Print model tensors";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            ctx.model().append(ChatMessage.system(
                    "Model tensors:\n" + String.join("\n", session.getConfig().getTensorNames())));
        }
    }
}
