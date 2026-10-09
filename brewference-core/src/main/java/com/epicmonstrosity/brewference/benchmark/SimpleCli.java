package com.epicmonstrosity.brewference.benchmark;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.model.ModelRunnerFactory;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.runtime.ModelSession;
import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.PromptTemplateRegistry;

import java.io.IOException;

public class SimpleCli {
    public static void main(final String[] args) throws IOException {
        final String modelFile = args[0];
        final GenerationOptions options = new GenerationOptions();

        final PromptTemplateRegistry promptRegistry = new PromptTemplateRegistry();
        final PromptTemplate promptTemplate = promptRegistry.get("jinja");
        final ChatTui chat = new ChatTui(promptTemplate, options, "You are a helpful assistant.");

        try (final ModelRunner modelRunner = ModelRunnerFactory.createModelRunner(modelFile, options, chat)) {
            final ModelSession activeSession = modelRunner.createSession();
            chat.setSession(activeSession, modelFile);
            promptTemplate.setPromptTemplate(activeSession.getConfig().getJinjaTemplate());
            promptTemplate.setRoleMapping(activeSession.getConfig().getRoleMapping());
            chat.run();
        }
    }
}
