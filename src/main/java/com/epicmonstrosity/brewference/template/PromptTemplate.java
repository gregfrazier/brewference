package com.epicmonstrosity.brewference.template;

import java.util.List;

public interface PromptTemplate {
    String id();
    String displayName();
    String render(List<ChatMessage> messages);
    String renderForCompletion(List<ChatMessage> messages);
}
