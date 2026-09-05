package com.epicmonstrosity.brewference.template.jinja;

import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.chat.ChatMessage;
import com.epicmonstrosity.brewference.template.chat.ChatRole;
import groovy.lang.GroovyShell;

import java.util.List;
import java.util.Map;

public final class JinjaGroovyRenderer implements PromptTemplate {
    private final JinjaTemplateCompiler compiler = new JinjaTemplateCompiler();
    private final JinjaBinding binding;
    private String promptTemplate;
    private Map<ChatRole, String> roleMapping;

    public JinjaGroovyRenderer() {
        this.binding = new JinjaBinding();

        // TODO: Gemma3 specific for now, need to allow other model types.
        binding.setVariable("bos_token", "<bos>");
        binding.setVariable("tools", null);
        binding.setVariable("add_generation_prompt", true);
        binding.setVariable("enable_thinking", true);
    }

    @Override
    public String id() {
        return "jinja";
    }

    @Override
    public String displayName() {
        return "Jinja to Groovy Renderer";
    }

    @Override
    public String render(final List<ChatMessage> messages) {
        if (promptTemplate == null) {
            throw new IllegalStateException("Prompt template not set");
        }

        final List<Map<String, String>> messageList = messages.stream().map(message ->
                Map.of(
                        "role", (roleMapping != null && roleMapping.containsKey(message.getRole()))
                                ? roleMapping.get(message.getRole())
                                : message.getRole().getModelName(),
                        "content", message.getContent()
                )).toList();
        binding.setVariable("messages", messageList);

        final String groovySource = compiler.compile(promptTemplate);
        final GroovyShell shell = new GroovyShell(JinjaGroovyRenderer.class.getClassLoader(), binding);

        //System.out.println(promptTemplate);
        //System.out.println(groovySource);

        return String.valueOf(shell.evaluate(groovySource));
    }

    @Override
    public String renderForCompletion(final List<ChatMessage> messages) {
        return render(messages);
    }

    @Override
    public PromptTemplate setPromptTemplate(final String promptTemplate) {
        this.promptTemplate = promptTemplate;
        return this;
    }

    @Override
    public PromptTemplate addBindingValue(final String key, final Object value) {
        binding.setVariable(key, value);
        return this;
    }

    @Override
    public PromptTemplate setRoleMapping(final Map<ChatRole, String> roleMapping) {
        this.roleMapping = roleMapping;
        return this;
    }
}
