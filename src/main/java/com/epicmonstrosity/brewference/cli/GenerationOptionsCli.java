package com.epicmonstrosity.brewference.cli;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import picocli.CommandLine.Option;

/**
 * Picocli mixin for populating GenerationOptions from CLI args.
 */
public class GenerationOptionsCli {

    @Option(names = {"-n", "--max-new-tokens"},
            description = "Maximum number of tokens to generate (default: ${DEFAULT-VALUE})")
    private int maxNewTokens = 4096;

    @Option(names = {"-t", "--temperature"},
            description = "Sampling temperature (default: ${DEFAULT-VALUE})")
    private float temperature = 1.0f;

    @Option(names = {"--top-p"},
            description = "Nucleus sampling probability mass (default: ${DEFAULT-VALUE})")
    private float topP = 0.95f;

    @Option(names = {"--top-k"},
            description = "Top-K sampling cutoff (default: ${DEFAULT-VALUE})")
    private int topK = 40;

    @Option(names = {"--repetition-penalty"},
            description = "Repetition penalty applied to logits (default: ${DEFAULT-VALUE})")
    private float repetitionPenalty = 1.0f;

    @Option(names = {"--echo-prompt"},
            description = "Echo the prompt back in the output (default: ${DEFAULT-VALUE})")
    private boolean echoPrompt = false;

    public GenerationOptions toGenerationOptions() {
        final GenerationOptions options = new GenerationOptions();
        options.setMaxNewTokens(maxNewTokens);
        options.setTemperature(temperature);
        options.setTopP(topP);
        options.setTopK(topK);
        options.setRepetitionPenalty(repetitionPenalty);
        options.setEchoPrompt(echoPrompt);

        return options;
    }
}
