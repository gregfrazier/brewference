package com.epicmonstrosity.brewference.template;

import com.epicmonstrosity.brewference.template.templates.ChatMLTemplate;
import com.epicmonstrosity.brewference.template.templates.GemmaTemplate;
import com.epicmonstrosity.brewference.template.templates.Llama2Template;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PromptTemplateRegistry {
    private final Map<String, PromptTemplate> templates = new LinkedHashMap<>();

    public PromptTemplateRegistry() {
        register(new ChatMLTemplate("qwen2", "Qwen 2.x / ChatML"));
        register(new ChatMLTemplate("smollm", "SmolLM / ChatML"));
        register(new ChatMLTemplate("phi3", "Phi3 / ChatML"));
        register(new GemmaTemplate());
        register(new Llama2Template());
    }

    public void register(final PromptTemplate template) {
        templates.put(template.id(), template);
    }

    public void register(final String forceId, final PromptTemplate template) {
        templates.put(forceId, template);
    }

    public PromptTemplate get(final String id) {
        final PromptTemplate template = templates.get(id);
        if (template == null) {
            throw new IllegalArgumentException("Unknown prompt template: " + id);
        }

        return template;
    }

    public List<PromptTemplate> all() {
        return new ArrayList<>(templates.values());
    }
}
