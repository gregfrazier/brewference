package com.epicmonstrosity.brewference.cli;

import picocli.CommandLine;

public class PromptOptionsCli {
    @CommandLine.Option(names = {"-p", "--prompt"},
            description = "System prompt text to feed the model", defaultValue = "You are a helpful assistant.")
    private String prompt = "You are a helpful assistant.";

    public String getPrompt() {
        return prompt;
    }
}
