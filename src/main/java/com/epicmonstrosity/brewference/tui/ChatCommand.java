package com.epicmonstrosity.brewference.tui;

@FunctionalInterface
public interface ChatCommand {
    /**
     * @param args the text following the command word (trimmed, possibly empty)
     * @param chat the owning chat loop, used to affect state (stop, clear, print, ...)
     */
    void execute(String args, ChatTui chat);
}
