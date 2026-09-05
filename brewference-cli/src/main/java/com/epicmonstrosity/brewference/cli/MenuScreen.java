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
import com.epicmonstrosity.tui.*;
import com.epicmonstrosity.tui.form.FormScreen;
import com.epicmonstrosity.tui.list.FilePickerScreen;
import com.epicmonstrosity.tui.overlay.Toast;
import com.epicmonstrosity.tui.screen.KeyBinding;
import com.epicmonstrosity.tui.screen.ScreenLayout;
import com.epicmonstrosity.tui.screen.ScreenTemplate;
import picocli.CommandLine;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;

public class MenuScreen extends ScreenTemplate {
    private UiHost host;
    private boolean invalidated = true;
    private int cursor;
    private String modelPath = "not loaded";

    private ModelOptionsCli modelOptionsCli;
    private PromptOptionsCli promptOptionsCli;
    private GenerationOptionsCli generationOptionsCli;
    private EditableGenerationOptions editableGenerationOptions;

    private record MenuEntry(String label, Supplier<String> hint, Runnable action) {
    }

    private final List<MenuEntry> entries;

    public MenuScreen(final ModelOptionsCli modelOptionsCli,
                      final PromptOptionsCli promptOptionsCli,
                      final GenerationOptionsCli generationOptionsCli) {
        this.modelOptionsCli = modelOptionsCli;
        this.promptOptionsCli = promptOptionsCli;
        this.generationOptionsCli = generationOptionsCli;

        this.editableGenerationOptions = EditableGenerationOptions.of(generationOptionsCli, modelOptionsCli);
        this.modelPath = modelOptionsCli.getGgufPath() == null ? "not loaded" : modelOptionsCli.getGgufPath();

        this.entries = List.of(
                new MenuEntry("Chat",
                        () -> isModelLoaded() ? "Start chat with " + modelPath : "No model loaded",
                        this::chatScreen),
                new MenuEntry("Load Model", () -> "Load model from filesystem", this::loadModel),
                new MenuEntry("Configuration", () -> "Modify configuration", this::configuration)
        );
    }

    @Override
    protected ScreenLayout buildLayout(final ViewContext ctx) {
        invalidated = false;
        final var body = new StringBuilder();
        for (int i = 0; i < entries.size(); i++) {
            final MenuEntry entry = entries.get(i);
            configField(body, cursor, i, entry.label(), entry.hint().get());
        }
        return new ScreenLayout(
                headerBordered("Brewference Main Menu", "", ctx.terminalCols() - 2),
                body.toString(),
                "",
                footerError("")
        );
    }

    @Override
    public void onEnter(final UiHost host) {
        this.host = host;
        this.invalidated = true;
    }

    @Override
    public void onExit() {

    }

    @Override
    public void onEvent(final UiMessage msg) {
        if (msg instanceof KeyPressed(final String key)) {
            handleInput(key);
            invalidated = true;
        }
    }

    @Override
    public boolean isInvalidated() {
        return invalidated;
    }

    @Override
    public List<KeyBinding> keyBindings() {
        return List.of(
                KeyBinding.primary("UP", "↑", "up", () -> cursor = Math.max(0, cursor - 1)),
                KeyBinding.primary("DOWN", "↓", "down", () -> cursor = Math.min(entries.size() - 1, cursor + 1)),
                KeyBinding.primary("ENTER", "Enter", "open", this::openSelected)
        );
    }

    private void openSelected() {
        if (host == null) {
            return;
        }
        entries.get(cursor).action().run();
    }

    private void loadModel() {
        host.navigation().push(FilePickerScreen.files(
                "Load GGUF Model",
                Path.of(System.getProperty("user.dir")),
                path -> {
                    host.showToast(Toast.success("Picked " + path));
                    this.modelPath = String.valueOf(path);
                }
        ));
    }

    private void configuration() {
        host.navigation().push(new FormScreen<>(
                "Configuration",
                EditableGenerationOptions.class,
                editableGenerationOptions,
                user -> host.showToast(Toast.success("Config updated"))
        ).popOnSubmit());
    }

    private boolean isModelLoaded() {
        return modelPath != null && !modelPath.equals("not loaded");
    }

    private void chatScreen() {
        if (!isModelLoaded()) {
            host.showToast(Toast.warning("Load a model first"));
            return;
        }

        final GenerationOptions options = editableGenerationOptions.toGenerationOptions();

        final PromptTemplateRegistry promptRegistry = new PromptTemplateRegistry();
        final PromptTemplate promptTemplate = promptRegistry.get(editableGenerationOptions.template);
        final ChatApp chat = new ChatApp(promptTemplate, options, promptOptionsCli.getPrompt());
        try  {
            final ModelRunner modelRunner = ModelRunnerFactory.createModelRunner(modelPath, options, chat);
            final ModelSession activeSession = modelRunner.createSession();
            promptTemplate.setPromptTemplate(activeSession.getConfig().getJinjaTemplate());
            promptTemplate.setRoleMapping(activeSession.getConfig().getRoleMapping());
            chat.setSession(activeSession, modelPath);
            chat.run(host, () -> {
                try {
                    modelRunner.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        } catch (IOException e) {
            host.showToast(Toast.error("Failed to load model: " + e.getMessage()));
        }
    }
}
