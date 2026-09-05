package com.epicmonstrosity.brewference.cli.screen.chat;

/**
 * Strategy for a chat slash command. Register instances on
 * {@link ChatCommands} or {@link ChatScreen#command(ChatCommand)}.
 */
public interface ChatCommand {
    String name();

    /**
     * Get the command description (for help display).
     *
     * @return The description, or empty string for none
     */
    default String description() {
        return "";
    }

    /**
     * Execute the command with the given context.
     *
     * @param ctx The command context containing input, model, and host
     */
    void execute(ChatCommandContext ctx);
}
