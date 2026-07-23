package com.epicmonstrosity.brewference.template.templates;

import com.epicmonstrosity.brewference.template.ChatMessage;
import com.epicmonstrosity.brewference.template.ChatRole;
import com.epicmonstrosity.brewference.template.PromptTemplate;

import java.util.List;

public class GemmaTemplate implements PromptTemplate {
    private static final String BOS_TOKEN = "<bos>";
    private static final String START_OF_TURN = "<start_of_turn>";
    private static final String END_OF_TURN = "<end_of_turn>";

    @Override
    public String id() {
        return "gemma";
    }

    @Override
    public String displayName() {
        return "Gemma Chat";
    }

    @Override
    public String render(final List<ChatMessage> messages) {
        final StringBuilder builder = new StringBuilder(BOS_TOKEN);

        for (final ChatMessage message : messages) {
            appendMessage(builder, message, true);
        }

        return builder.toString();
    }

    @Override
    public String renderForCompletion(final List<ChatMessage> messages) {
        final StringBuilder builder = new StringBuilder(render(messages));

        builder.append(START_OF_TURN)
                .append("model")
                .append('\n');

        return builder.toString();
    }

    private void appendMessage(final StringBuilder builder, final ChatMessage message, final boolean closeTurn) {
        builder.append(START_OF_TURN)
                .append(toGemmaRole(message.getRole()))
                .append('\n')
                .append(message.getContent());

        if (closeTurn) {
            builder.append(END_OF_TURN)
                    .append('\n');
        }
    }

    private String toGemmaRole(final ChatRole role) {
        switch (role) {
            case USER:
            case SYSTEM: return "user";
            case MODEL: return "model";
        }
        return null;
    }
}
