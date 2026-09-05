package com.epicmonstrosity.brewference.cli;

import com.epicmonstrosity.brewference.cli.option.GenerationOptionsCli;
import com.epicmonstrosity.brewference.cli.option.ModelOptionsCli;
import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.tui.form.FormField;

public class EditableGenerationOptions {
    @FormField(label = "Context Length", required = true, order = 1, help = "Default 4096")
    public int contextLength = 4096;
    @FormField(label = "maxNewTokens", required = true, order = 2, help = "Default 4096")
    public int maxNewTokens = 4096;
    @FormField(label = "temperature", required = true, order = 3, help = "Default 1.0")
    public float temperature = 1.0f;
    @FormField(label = "topP", required = true, order = 4, help = "Default 0.95")
    public float topP = 0.95f;
    @FormField(label = "topK", required = true, order = 5, help = "Default 40")
    public int topK = 40;
    @FormField(label = "repetitionPenalty", required = true, order = 6, help = "Default 1.0")
    public float repetitionPenalty = 1.0f;
    @FormField(label = "echoPrompt", required = true, order = 7, help = "Default false")
    public boolean echoPrompt = false;
    @FormField(label = "disableThinking", required = true, order = 8, help = "Default false")
    public boolean disableThinking = false;
    @FormField(label = "bosToken", order = 9, help = "Beginning of sentence token")
    public int bosToken;
    @FormField(label = "eosToken", order = 10, help = "End of sentence token")
    public int eosToken;
    @FormField(label = "Template Id", required = true, order = 1, help = "Default jinja")
    public String template = "jinja";

    public static EditableGenerationOptions of(final GenerationOptionsCli generationOptionsCli,
                                               final ModelOptionsCli modelOptionsCli) {
        final var options = new EditableGenerationOptions();
        final GenerationOptions generationOptions = generationOptionsCli.toGenerationOptions();
        options.contextLength = generationOptions.getContextLength();
        options.maxNewTokens = generationOptions.getMaxNewTokens();
        options.temperature = generationOptions.getTemperature();
        options.topP = generationOptions.getTopP();
        options.topK = generationOptions.getTopK();
        options.repetitionPenalty = generationOptions.getRepetitionPenalty();
        options.echoPrompt = generationOptions.isEchoPrompt();
        options.disableThinking = generationOptions.isDisableThinking();
        options.template = modelOptionsCli.getTemplateId();
        //options.bosToken = generationOptions.getBosToken();
        //options.eosToken = generationOptions.getEosToken();
        return options;
    }

    public GenerationOptions toGenerationOptions() {
        final GenerationOptions options = new GenerationOptions();
        options.setMaxNewTokens(maxNewTokens);
        options.setTemperature(temperature);
        options.setTopP(topP);
        options.setTopK(topK);
        options.setRepetitionPenalty(repetitionPenalty);
        options.setEchoPrompt(echoPrompt);
        options.setContextLength(contextLength);
        options.setDisableThinking(disableThinking);
        return options;
    }
}
