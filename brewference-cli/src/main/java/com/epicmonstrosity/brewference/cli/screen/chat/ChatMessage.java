package com.epicmonstrosity.brewference.cli.screen.chat;

import java.time.Instant;

/**
 * A chat message with role, text content, and timestamp.
 */
public record ChatMessage(String role, String text, Instant timestamp, boolean streaming) {
    /** User message role. */
    public static final String USER = "user";
    /** Assistant message role. */
    public static final String ASSISTANT = "assistant";
    /** System message role. */
    public static final String SYSTEM = "system";
    /** Tool message role. */
    public static final String TOOL = "tool";

    /**
     * Create a chat message with the current timestamp.
     *
     * @param role The message role
     * @param text The message text
     */
    public ChatMessage(final String role, final String text) {
        this(role, text, Instant.now(), false);
    }

    /**
     * Create a new message with updated text.
     *
     * @param next The new text
     * @return A new ChatMessage with the updated text
     */
    public ChatMessage withText(final String next) {
        return new ChatMessage(role, next == null ? "" : next, timestamp, streaming);
    }

    /**
     * Mark this message as finished (stop streaming).
     *
     * @return A new ChatMessage with streaming set to false
     */
    public ChatMessage finished() {
        return new ChatMessage(role, text, timestamp, false);
    }

    /**
     * Create a user message.
     *
     * @param text The message text
     * @return A new ChatMessage with USER role
     */
    public static ChatMessage user(final String text) {
        return new ChatMessage(USER, text);
    }

    /**
     * Create an assistant message.
     *
     * @param text The message text
     * @return A new ChatMessage with ASSISTANT role
     */
    public static ChatMessage assistant(final String text) {
        return new ChatMessage(ASSISTANT, text);
    }

    /**
     * Create a system message.
     *
     * @param text The message text
     * @return A new ChatMessage with SYSTEM role
     */
    public static ChatMessage system(final String text) {
        return new ChatMessage(SYSTEM, text);
    }

    /**
     * Create a tool message.
     *
     * @param text The message text
     * @return A new ChatMessage with TOOL role
     */
    public static ChatMessage tool(final String text) {
        return new ChatMessage(TOOL, text);
    }
}
