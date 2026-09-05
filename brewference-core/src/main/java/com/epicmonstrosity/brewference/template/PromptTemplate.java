package com.epicmonstrosity.brewference.template;

import com.epicmonstrosity.brewference.template.chat.ChatMessage;
import com.epicmonstrosity.brewference.template.chat.ChatRole;

import java.util.List;
import java.util.Map;

public interface PromptTemplate {
    String id();
    String displayName();
    String render(List<ChatMessage> messages);
    String renderForCompletion(List<ChatMessage> messages);

    // Cheap way of not having to implement this in every template
    default PromptTemplate setPromptTemplate(String promptTemplate) { return this; }
    default PromptTemplate addBindingValue(String key, Object value) { return this; }
    default PromptTemplate setRoleMapping(Map<ChatRole, String> roleMapping) { return this;}
}
