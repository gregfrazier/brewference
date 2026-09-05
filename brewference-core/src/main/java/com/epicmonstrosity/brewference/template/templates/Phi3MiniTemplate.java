package com.epicmonstrosity.brewference.template.templates;

import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.chat.ChatMessage;
import com.epicmonstrosity.brewference.template.chat.ChatRole;

import java.util.List;

public class Phi3MiniTemplate implements PromptTemplate {
    private static final String BOS_TOKEN = "<s>";
    private final String id;
    private final String displayName;

    public Phi3MiniTemplate(final String id, final String displayName) {
        this.id = id;
        this.displayName = displayName;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String displayName() {
        return displayName;
    }

    @Override
    public String render(final List<ChatMessage> messages) {
        final StringBuilder builder = new StringBuilder(BOS_TOKEN);

        for (final ChatMessage message : messages) {
            builder.append("<|")
                    .append(toTemplateRole(message.getRole()))
                    .append("|>")
                    .append('\n')
                    .append(message.getContent())
                    .append("<|end|>\n");
        }

        return builder.toString();
    }

    @Override
    public String renderForCompletion(final List<ChatMessage> messages) {
        return render(messages) + "<|assistant|>\n";
    }

    private String toTemplateRole(final ChatRole role) {
        return switch (role) {
            case SYSTEM -> "system";
            case USER -> "user";
            case MODEL -> "assistant";
        };
    }
}
