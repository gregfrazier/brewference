package com.epicmonstrosity.brewference.cli.screen.chat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Named slash-command registry. {@link #standard()} includes {@code /help}.
 */
public final class ChatCommands {
    private final Map<String, ChatCommand> byName = new LinkedHashMap<>();

    /**
     * Get a standard command registry with the /help command registered.
     *
     * @return A new ChatCommands instance with /help
     */
    public static ChatCommands standard() {
        return new ChatCommands().register(new HelpChatCommand());
    }

    /**
     * Register a chat command.
     *
     * @param command The command to register
     * @return This ChatCommands instance for chaining
     * @throws IllegalArgumentException If the command is null or has no name
     */
    public synchronized ChatCommands register(final ChatCommand command) {
        if (command == null || command.name() == null || command.name().isBlank()) {
            throw new IllegalArgumentException("Chat command needs a name");
        }
        byName.put(normalize(command.name()), command);
        return this;
    }

    /**
     * Find a command by name.
     *
     * @param name The command name (without /)
     * @return Optional containing the command if found, empty otherwise
     */
    public synchronized Optional<ChatCommand> find(final String name) {
        if (name == null || name.isBlank()) {
            return Optional.empty();
        }
        return Optional.ofNullable(byName.get(normalize(name)));
    }

    /**
     * Get all registered commands.
     *
     * @return Unmodifiable list of all commands
     */
    public synchronized List<ChatCommand> all() {
        return List.copyOf(byName.values());
    }

    /**
     * Dispatch a slash command to the appropriate handler.
     *
     * @param input The input string (should start with /)
     * @param ctx The command context
     * @return true if the input was a command (handled or unknown), false otherwise
     */
    public boolean dispatch(final String input, final ChatCommandContext ctx) {
        final Optional<Invocation> invocation = parse(input);
        if (invocation.isEmpty()) {
            return false;
        }
        final Invocation found = invocation.get();
        final ChatCommand command = find(found.name()).orElse(null);
        final ChatCommandContext resolved = new ChatCommandContext(
                found.raw(),
                found.name(),
                found.args(),
                ctx == null ? null : ctx.model(),
                ctx == null ? null : ctx.host(),
                this
        );
        if (command == null) {
            if (resolved.model() != null) {
                resolved.model().append(ChatMessage.system(
                        "Unknown command /" + found.name() + ". Try /help."
                ));
            }
            return true;
        }
        command.execute(resolved);
        return true;
    }

    /**
     * Parse a slash command input into name and args.
     *
     * @param input The input string to parse
     * @return Optional containing Invocation if valid, empty otherwise
     */
    static Optional<Invocation> parse(final String input) {
        if (input == null) {
            return Optional.empty();
        }
        final String trimmed = input.trim();
        if (trimmed.length() < 2 || trimmed.charAt(0) != '/') {
            return Optional.empty();
        }
        if (Character.isWhitespace(trimmed.charAt(1))) {
            return Optional.empty();
        }
        int split = -1;
        for (int i = 1; i < trimmed.length(); i++) {
            if (Character.isWhitespace(trimmed.charAt(i))) {
                split = i;
                break;
            }
        }
        final String name = split < 0 ? trimmed.substring(1) : trimmed.substring(1, split);
        if (name.isEmpty()) {
            return Optional.empty();
        }
        final String args = split < 0 ? "" : trimmed.substring(split).trim();
        return Optional.of(new Invocation(trimmed, name, args));
    }

    /**
     * Normalize a command name to lowercase trimmed form.
     *
     * @param name The command name to normalize
     * @return The normalized name
     */
    private static String normalize(final String name) {
        return name.trim().toLowerCase(Locale.ROOT);
    }

    /**
     * A parsed slash command invocation.
     */
    record Invocation(String raw, String name, String args) {
    }
}
