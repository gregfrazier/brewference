package com.epicmonstrosity.brewference.template.templates;

import com.epicmonstrosity.brewference.template.ChatMessage;
import com.epicmonstrosity.brewference.template.ChatRole;
import com.epicmonstrosity.brewference.template.PromptTemplate;

import java.util.List;

public class ChatMLTemplate implements PromptTemplate {
    private final String id;
    private final String displayName;

    public ChatMLTemplate(final String id, final String displayName) {
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
        final StringBuilder builder = new StringBuilder();

        for (final ChatMessage message : messages) {
            builder.append("<|im_start|>")
                    .append(toTemplateRole(message.getRole()))
                    .append('\n')
                    .append(message.getContent())
                    .append("<|im_end|>\n");
        }

        return builder.toString();
    }

    @Override
    public String renderForCompletion(final List<ChatMessage> messages) {
        return render(messages) + "<|im_start|>assistant\n";
    }

    private String toTemplateRole(final ChatRole role) {
        switch (role) {
            case SYSTEM: return "system";
            case USER: return "user";
            case MODEL: return "assistant";
        }
        return null;
    }
}
