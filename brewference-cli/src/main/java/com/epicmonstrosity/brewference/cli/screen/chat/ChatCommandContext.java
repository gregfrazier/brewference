package com.epicmonstrosity.brewference.cli.screen.chat;

import com.epicmonstrosity.tui.UiHost;

/**
 * Context object passed to chat command execution.
 * Contains the raw input, parsed name/args, chat model, and UI host.
 */
public final class ChatCommandContext {
    private final String raw;
    private final String name;
    private final String args;
    private final ChatModel model;
    private final UiHost host;
    private final ChatCommands commands;

    /**
     * Create a new command context.
     *
     * @param raw The full raw input string
     * @param name The parsed command name (without /)
     * @param args The parsed command arguments
     * @param model The chat model
     * @param host The UI host
     * @param commands The command registry
     */
    public ChatCommandContext(final String raw,
                              final String name,
                              final String args,
                              final ChatModel model,
                              final UiHost host,
                              final ChatCommands commands) {
        this.raw = raw == null ? "" : raw;
        this.name = name == null ? "" : name;
        this.args = args == null ? "" : args;
        this.model = model;
        this.host = host;
        this.commands = commands;
    }

    /**
     * Get the full raw input string.
     *
     * @return The raw input
     */
    public String raw() {
        return raw;
    }

    /**
     * Get the parsed command name (without leading /).
     *
     * @return The command name
     */
    public String name() {
        return name;
    }

    /**
     * Get the parsed command arguments.
     *
     * @return The arguments
     */
    public String args() {
        return args;
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
     * Get the UI host.
     *
     * @return The UI host
     */
    public UiHost host() {
        return host;
    }

    /**
     * Get the command registry.
     *
     * @return The command registry
     */
    public ChatCommands commands() {
        return commands;
    }
}
