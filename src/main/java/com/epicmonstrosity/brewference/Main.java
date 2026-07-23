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
//        if (args.length < 1) {
//            System.out.println("Usage: java -jar brewference.jar <model.gguf> [templateId] [systemPrompt]");
//            System.out.println("  templateId: one of qwen2, smollm, phi3, gemma, llama2 (default: llama2)");
//            return;
//        }
//
//        final String modelPath = args[0];
//        final String templateId = args.length > 1 ? args[1] : "qwen2";
//        final String systemPrompt = args.length > 2 ? args[2] : "You are a helpful assistant.";

//        final PromptTemplateRegistry promptRegistry = new PromptTemplateRegistry();
//        final PromptTemplate promptTemplate = promptRegistry.get(templateId);
//
//        final GenerationOptions options = new GenerationOptions().setMaxNewTokens(4096);
//
//        // ChatTui is the TokenConsumer: generation streams decoded tokens back through it, and it
//        // also receives model-load debug output during createModelRunner(...).
//        final ChatTui chat = new ChatTui(promptTemplate, options, systemPrompt);
//
//        final ModelRunner modelRunner = ModelRunnerFactory.createModelRunner(modelPath, chat);
//        chat.setSession(modelRunner.createSession());
//
//        chat.run();
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
