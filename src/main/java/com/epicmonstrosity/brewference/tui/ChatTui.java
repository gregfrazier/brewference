package com.epicmonstrosity.brewference.tui;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.generation.GenerationResult;
import com.epicmonstrosity.brewference.generation.TokenConsumer;
import com.epicmonstrosity.brewference.runtime.ModelSession;
import com.epicmonstrosity.brewference.template.ChatConversation;
import com.epicmonstrosity.brewference.template.ChatMessage;
import com.epicmonstrosity.brewference.template.PromptTemplate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A minimal console chat loop.
 */
public final class ChatTui implements TokenConsumer {
    private final PromptTemplate template;
    private final GenerationOptions options;
    private final String systemPrompt;
    private final Map<String, ChatCommand> commands = new LinkedHashMap<>();
    private final Map<String, String> commandDescriptions = new LinkedHashMap<>();
    private ModelSession session;
    private ChatConversation conversation;
    private final StringBuilder currentReply = new StringBuilder();
    private boolean running = true;
    private boolean chatting = false;
    private String filename;

    public ChatTui(final PromptTemplate template,
                   final GenerationOptions options,
                   final String systemPrompt) {
        this.template = template;
        this.options = options;
        this.systemPrompt = systemPrompt;
        this.conversation = newConversation();
        registerCommands();
    }

    public void setSession(final ModelSession session, final String filename) {
        this.session = session;
        this.filename = filename;
    }

    private void registerCommands() {
        register("exit", "Quit the chat", (args, chat) -> chat.stop());
        register("clear", "Reset the conversation", (args, chat) -> chat.clearContext());
        register("help", "List available commands", (args, chat) -> chat.printHelp());
        register("details", "Print model details", (args, chat) -> chat.printModelDetails());
        register("metadata", "Print model metadata", (args, chat) -> chat.printModelMetadata());
    }

    private void register(final String name, final String description, final ChatCommand handler) {
        commands.put(name, handler);
        commandDescriptions.put(name, description);
    }

    public void run() {
        if (session == null) {
            throw new IllegalStateException("setSession(...) must be called before run()");
        }

        chatting = true;
        boolean systemPrompted = false;
        printBanner();

        final BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        while (running) {
            System.out.print("> ");
            System.out.flush();

            final String line;
            try {
                line = reader.readLine();
            } catch (final IOException e) {
                throw new UncheckedIOException(e);
            }

            if (line == null) {
                break;
            }

            final String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                continue;
            }

            if (trimmed.startsWith("/")) {
                dispatchCommand(trimmed);
            } else {
                chatTurn(line, systemPrompted);
                if (!systemPrompted)
                    systemPrompted = true;
            }
        }

        System.out.println();
        System.out.println("Goodbye.");
    }

    private void dispatchCommand(final String input) {
        final String body = input.substring(1);
        final int space = body.indexOf(' ');
        final String name = (space < 0 ? body : body.substring(0, space)).toLowerCase();
        final String args = space < 0 ? "" : body.substring(space + 1).trim();

        final ChatCommand command = commands.get(name);
        if (command == null) {
            System.out.println("Unknown command: /" + name + " — type /help");
            return;
        }

        command.execute(args, this);
    }

    /** Runs a single user turn: send the message, stream the reply, remember it. */
    private void chatTurn(final String message, final boolean systemPrompted) {
        final ChatMessage chatMessage = conversation.addUserMessage(message);

        final String prompt = template.renderForCompletion(
                systemPrompted ? Collections.singletonList(chatMessage) : conversation.messages()
        );

        currentReply.setLength(0);
        session.generate(prompt, options);

        conversation.addAssistantMessage(currentReply.toString());
    }

    public void stop() {
        running = false;
    }

    public void clearContext() {
        conversation = newConversation();
        session.clear(); // no-op
        System.out.println("(context cleared)");
    }

    /** Prints the list of registered slash commands. */
    public void printHelp() {
        System.out.println("Commands:");
        for (final Map.Entry<String, String> entry : commandDescriptions.entrySet()) {
            System.out.printf("  /%-8s %s%n", entry.getKey(), entry.getValue());
        }
    }

    public void printModelDetails() {
        System.out.println("Model details:");
        System.out.println(session.getConfig().toString());
    }

    public void printModelMetadata() {
        System.out.println("Model metadata:");
        System.out.println(session.getConfig().getMetadata().toString());
    }

    private ChatConversation newConversation() {
        final ChatConversation fresh = new ChatConversation();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            fresh.addSystemMessage(systemPrompt);
        }
        return fresh;
    }

    private void printBanner() {
        System.out.println();
        System.out.println("Brewference chat - " + filename);
        System.out.println("Type a message and press Enter.");
        System.out.println("Slash commands: /exit, /clear, /help");
        System.out.println();
    }

    @Override
    public void onPrefillToken(final int position, final int tokenId, final String tokenText) {
        if (options.isEchoPrompt()) {
            System.out.printf("tokenId: %d tokenText: %s%n", tokenId, tokenText);
            System.out.flush();
        }
    }

    @Override
    public void onGeneratedToken(final int position, final int tokenId, final String tokenText) {
        System.out.print(tokenText);
        System.out.flush();
        currentReply.append(tokenText);
    }

    @Override
    public void onComplete(final GenerationResult result) {
        System.out.println();
        System.out.printf("[generated: %d tokens, eos: %s, elapsedMs: %s]%n",
                result.getGeneratedTokenCount(), result.isStoppedByEos(), result.getElapsedNanos() / 1_000_000.0);
    }

    @Override
    public void onDebug(final String debug) {
        if (!chatting) {
            System.out.println(debug);
        }
    }
}
