package com.epicmonstrosity.brewference.template;

import java.util.ArrayList;
import java.util.List;

public class ChatConversation {
    private final List<ChatMessage> messages = new ArrayList<>();

    public void addSystemMessage(final String content) {
        messages.add(new ChatMessage(ChatRole.SYSTEM, content));
    }

    public ChatMessage addUserMessage(final String content) {
        final ChatMessage chatMessage = new ChatMessage(ChatRole.USER, content);
        messages.add(chatMessage);

        return chatMessage;
    }

    public void addAssistantMessage(final String content) {
        messages.add(new ChatMessage(ChatRole.MODEL, content));
    }

    public List<ChatMessage> messages() {
        return new ArrayList<>(messages);
    }
}
