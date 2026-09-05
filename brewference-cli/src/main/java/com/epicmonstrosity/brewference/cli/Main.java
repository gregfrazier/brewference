package com.epicmonstrosity.brewference.cli;

import com.epicmonstrosity.brewference.cli.option.GenerationOptionsCli;
import com.epicmonstrosity.brewference.cli.option.ModelOptionsCli;
import com.epicmonstrosity.brewference.cli.option.PromptOptionsCli;
import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.model.ModelRunnerFactory;
import com.epicmonstrosity.brewference.runtime.ModelRunner;
import com.epicmonstrosity.brewference.runtime.ModelSession;
import com.epicmonstrosity.brewference.template.PromptTemplate;
import com.epicmonstrosity.brewference.template.PromptTemplateRegistry;
import com.epicmonstrosity.tui.TuiApp;
import com.epicmonstrosity.tui.ansi.Theme;
import picocli.CommandLine;
import picocli.CommandLine.Command;

import java.util.concurrent.Callable;

@Command(name = "brewference", description = "lightweight inference in java")
public class Main implements Callable<Integer> {
    @CommandLine.Mixin
    private ModelOptionsCli modelOptionsCli;
    @CommandLine.Mixin
    private PromptOptionsCli promptOptionsCli;
    @CommandLine.Mixin
    private GenerationOptionsCli generationOptionsCli;

    public static void main(final String[] args) {
        final int exitCode = new CommandLine(new Main()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        try (final TuiApp app = TuiApp.open()) {
            app.theme(Theme.DEFAULT);
            app.onGlobalKey("CTRL_C", TuiApp.Signal.EXIT);
            app.run(new MenuScreen(modelOptionsCli, promptOptionsCli, generationOptionsCli));
        }
        return 0;
    }
}
