package com.epicmonstrosity.brewference.cli;

import picocli.CommandLine.Option;

public class ModelOptionsCli {

    @Option(names = {"-m", "--model"}, required = true,
            description = "Path to the GGUF model file")
    private String ggufPath;

    @Option(names = {"--context-length"},
            description = "Context window size (default: ${DEFAULT-VALUE})")
    private int contextLength = 4096;

    @Option(names = {"--template"}, required = true,
            description = "Template ID to use for generation")
    private String templateId;

    public String getGgufPath() {
        return ggufPath;
    }

    public int getContextLength() {
        return contextLength;
    }

    public String getTemplateId() {
        return templateId;
    }
}
