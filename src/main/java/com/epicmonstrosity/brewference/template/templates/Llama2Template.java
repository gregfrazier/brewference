package com.epicmonstrosity.brewference.template.templates;

import com.epicmonstrosity.brewference.template.ChatMessage;
import com.epicmonstrosity.brewference.template.ChatRole;
import com.epicmonstrosity.brewference.template.PromptTemplate;

import java.util.ArrayList;
import java.util.List;

public class Llama2Template implements PromptTemplate {
    private static final String BOS_TOKEN = "<s>";
    private static final String EOS_TOKEN = "</s>";
    private static final String INST_START = "[INST]";
    private static final String INST_END = "[/INST]";
    private static final String SYS_START = "<<SYS>>";
    private static final String SYS_END = "<</SYS>>";

    private static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant.";

    @Override
    public String id() {
        return "llama2";
    }

    @Override
    public String displayName() {
        return "Llama 2 Chat";
    }

    @Override
    public String render(final List<ChatMessage> messages) {
        final ParsedMessages parsedMessages = parseMessages(messages);
        return renderConversation(parsedMessages, false);
    }

    @Override
    public String renderForCompletion(final List<ChatMessage> messages) {
        final ParsedMessages parsedMessages = parseMessages(messages);
        return renderConversation(parsedMessages, true);
    }

    private String renderConversation(final ParsedMessages parsedMessages, final boolean forCompletion) {
        final StringBuilder builder = new StringBuilder();

        final List<ChatMessage> conversationMessages = parsedMessages.getConversationMessages();

        if (conversationMessages.isEmpty()) {
            builder.append(BOS_TOKEN)
                    .append(INST_START)
                    .append(' ')
                    .append(renderSystemPrompt(parsedMessages.getSystemPrompt()))
                    .append(INST_END);

            return builder.toString();
        }

        int index = 0;
        boolean isFirstUserTurn = true;

        while (index < conversationMessages.size()) {
            final ChatMessage userMessage = conversationMessages.get(index);

            if (userMessage.getRole() != ChatRole.USER) {
                throw new IllegalArgumentException("Llama 2 chat format expects user and assistant messages to alternate.");
            }

            builder.append(BOS_TOKEN)
                    .append(INST_START)
                    .append(' ');

            if (isFirstUserTurn) {
                builder.append(renderSystemPrompt(parsedMessages.getSystemPrompt()));
                isFirstUserTurn = false;
            }

            builder.append(userMessage.getContent())
                    .append(' ')
                    .append(INST_END);

            index++;

            if (index < conversationMessages.size()) {
                final ChatMessage assistantMessage = conversationMessages.get(index);

                if (assistantMessage.getRole() != ChatRole.MODEL) {
                    throw new IllegalArgumentException("Llama 2 chat format expects assistant response after user message.");
                }

                builder.append(' ')
                        .append(assistantMessage.getContent())
                        .append(' ')
                        .append(EOS_TOKEN);

                index++;
            } else if (!forCompletion) {
                builder.append(' ')
                        .append(EOS_TOKEN);
            }
        }

        return builder.toString();
    }

    private String renderSystemPrompt(final String systemPrompt) {
        return SYS_START + "\n"
                + systemPrompt + "\n"
                + SYS_END + "\n\n";
    }

    private ParsedMessages parseMessages(final List<ChatMessage> messages) {
        String systemPrompt = DEFAULT_SYSTEM_PROMPT;
        final List<ChatMessage> conversationMessages = new ArrayList<>();

        for (final ChatMessage message : messages) {
            if (message.getRole() == ChatRole.SYSTEM) {
                systemPrompt = message.getContent();
            } else {
                conversationMessages.add(message);
            }
        }

        return new ParsedMessages(systemPrompt, conversationMessages);
    }

    private static class ParsedMessages {
        private final String systemPrompt;
        private final List<ChatMessage> conversationMessages;

        public ParsedMessages(final String systemPrompt, final List<ChatMessage> conversationMessages) {
            this.systemPrompt = systemPrompt;
            this.conversationMessages = conversationMessages;
        }

        public String getSystemPrompt() {
            return systemPrompt;
        }

        public List<ChatMessage> getConversationMessages() {
            return conversationMessages;
        }
    }
}
