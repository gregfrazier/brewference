package com.epicmonstrosity.brewference.tui;

/**
 * A slash command handled by the {@link ChatTui} loop (e.g. {@code /exit}, {@code /clear}).
 *
 * <p>Register new commands in {@link ChatTui#registerCommands()} with a single call to
 * {@code register(name, description, handler)} — that is the intended extension point.
 */
@FunctionalInterface
public interface ChatCommand {

    /**
     * @param args the text following the command word (trimmed, possibly empty)
     * @param chat the owning chat loop, used to affect state (stop, clear, print, ...)
     */
    void execute(String args, ChatTui chat);
}
