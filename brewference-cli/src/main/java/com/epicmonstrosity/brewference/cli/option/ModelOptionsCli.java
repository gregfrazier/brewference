package com.epicmonstrosity.brewference.cli.option;

import picocli.CommandLine.Option;

public class ModelOptionsCli {

    @Option(names = {"-m", "--model"},
            description = "Path to the GGUF model file")
    private String ggufPath;


    @Option(names = {"--template"},
            description = "Template ID to use for generation (default: ${DEFAULT-VALUE})")
    private String templateId = "jinja";

    public String getGgufPath() {
        return ggufPath;
    }

    public String getTemplateId() {
        return templateId;
    }
}
