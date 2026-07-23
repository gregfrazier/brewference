package com.epicmonstrosity.brewference;

import com.epicmonstrosity.brewference.cli.GenerationOptionsCli;
import com.epicmonstrosity.brewference.cli.ModelOptionsCli;
import com.epicmonstrosity.brewference.cli.PromptOptionsCli;
import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.model.ModelRunnerFactory;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.PromptTemplateRegistry;
import com.epicmonstrosity.brewference.tui.ChatTui;
import picocli.CommandLine;
import picocli.CommandLine.Command;

import java.io.IOException;
import java.util.concurrent.Callable;

@Command(name = "brewference", description = "lightweight inference in java")
public class Main implements Callable<Integer> {
    @CommandLine.Mixin
    private ModelOptionsCli modelOptionsCli;
    @CommandLine.Mixin
    private PromptOptionsCli promptOptionsCli;
    @CommandLine.Mixin
    private GenerationOptionsCli generationOptionsCli;

    public static void main(final String[] args) throws IOException {
        final int exitCode = new CommandLine(new Main()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        final GenerationOptions options = generationOptionsCli.toGenerationOptions();

        final PromptTemplateRegistry promptRegistry = new PromptTemplateRegistry();
        final PromptTemplate promptTemplate = promptRegistry.get(modelOptionsCli.getTemplateId());
        final ChatTui chat = new ChatTui(promptTemplate, options, promptOptionsCli.getPrompt());

        final ModelRunner modelRunner = ModelRunnerFactory.createModelRunner(modelOptionsCli.getGgufPath(), chat);
        chat.setSession(modelRunner.createSession(), modelOptionsCli.getGgufPath());

        chat.run();


        return 0;
    }
}
