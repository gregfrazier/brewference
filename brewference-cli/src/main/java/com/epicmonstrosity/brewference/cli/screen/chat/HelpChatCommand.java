package com.epicmonstrosity.brewference.cli.screen.chat;

/**
 * Built-in {@code /help} command. Lists every command on the current registry.
 */
public final class HelpChatCommand implements ChatCommand {
    public static final String NAME = "help";

    /**
     * Get the command name.
     *
     * @return The command name
     */
    @Override
    public String name() {
        return NAME;
    }

    /**
     * Get the command description.
     *
     * @return The description
     */
    @Override
    public String description() {
        return "List chat commands";
    }

    /**
     * Execute the help command by listing all registered commands.
     *
     * @param ctx The command context
     */
    @Override
    public void execute(final ChatCommandContext ctx) {
        final StringBuilder text = new StringBuilder("Commands:");
        if (ctx.commands() != null) {
            for (final ChatCommand command : ctx.commands().all()) {
                text.append('\n').append('/').append(command.name());
                if (command.description() != null && !command.description().isBlank()) {
                    text.append(" — ").append(command.description());
                }
            }
        }
        if (ctx.model() != null) {
            ctx.model().append(ChatMessage.system(text.toString()));
        }
    }
}
