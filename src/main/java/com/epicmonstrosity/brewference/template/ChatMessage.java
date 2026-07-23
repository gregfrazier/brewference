package com.epicmonstrosity.brewference.template;

public class ChatMessage {
    private final ChatRole role;
    private final String content;

    public ChatMessage(final ChatRole role, final String content) {
        this.role = role;
        this.content = content;
    }

    public ChatRole getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }
}
