package com.epicmonstrosity.brewference.cli.screen.chat;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatCommandsTest {
    @Test
    void helpListsRegisteredCommands() {
        final ChatModel model = new ChatModel();
        final ChatCommands commands = ChatCommands.standard().register(new EchoCommand());
        final ChatCommandContext ctx = new ChatCommandContext("/help", "help", "", model, null, commands);

        assertTrue(commands.dispatch("/help", ctx));
        final String text = model.messages().getLast().text();
        assertTrue(text.contains("/help"));
        assertTrue(text.contains("/echo"));
        assertEquals(ChatMessage.SYSTEM, model.messages().getLast().role());
    }

    @Test
    void unknownCommandStaysInChatAndIsNotASubmit() {
        final ChatModel model = new ChatModel();
        final ChatCommands commands = ChatCommands.standard();
        assertTrue(commands.dispatch("/nope extra", new ChatCommandContext("/nope extra", "nope", "extra", model, null, commands)));
        assertTrue(model.messages().getLast().text().contains("Unknown command /nope"));
        assertTrue(model.messages().getLast().text().contains("/help"));
    }

    @Test
    void customCommandReceivesArgs() {
        final ChatModel model = new ChatModel();
        final AtomicReference<String> seen = new AtomicReference<>();
        final ChatCommands commands = ChatCommands.standard().register(new ChatCommand() {
            @Override
            public String name() {
                return "echo";
            }

            @Override
            public String description() {
                return "Repeat arguments";
            }

            @Override
            public void execute(final ChatCommandContext ctx) {
                seen.set(ctx.args());
                ctx.model().append(ChatMessage.system(ctx.args()));
            }
        });

        assertTrue(commands.dispatch("/echo hello there", context(model, commands)));
        assertEquals("hello there", seen.get());
        assertEquals("hello there", model.messages().getLast().text());
    }

    @Test
    void ordinaryTextIsNotACommand() {
        final ChatCommands commands = ChatCommands.standard();
        final ChatModel model = new ChatModel();
        assertFalse(commands.dispatch("hello /help", context(model, commands)));
        assertFalse(commands.dispatch("/", context(model, commands)));
        assertFalse(commands.dispatch("/ help", context(model, commands)));
        assertTrue(model.messages().isEmpty());
    }

    @Test
    void screenRoutesSlashCommandsAndNotSubmit() {
        final ChatModel model = new ChatModel();
        final List<String> submitted = new ArrayList<>();
        final ChatScreen screen = new ChatScreen("Demo", model, submitted::add, () -> {});
        model.type("/help");
        screen.onEvent(new com.epicmonstrosity.tui.KeyPressed("ENTER"));

        assertTrue(submitted.isEmpty());
        assertTrue(model.messages().getLast().text().contains("/help"));
        assertEquals("", model.inputBuffer());
    }

    @Test
    void screenSubmitsOrdinaryTextIncludingQuestionMark() {
        final ChatModel model = new ChatModel();
        final List<String> submitted = new ArrayList<>();
        final ChatScreen screen = new ChatScreen("Demo", model, submitted::add, () -> {});
        assertTrue(screen.capturesGlobalKey("?"));
        assertFalse(screen.capturesGlobalKey("FN_1"));

        model.type("what?");
        screen.onEvent(new com.epicmonstrosity.tui.KeyPressed("ENTER"));
        assertEquals(List.of("what?"), submitted);
        assertEquals(ChatMessage.USER, model.messages().getLast().role());
    }

    private static ChatCommandContext context(final ChatModel model, final ChatCommands commands) {
        return new ChatCommandContext("", "", "", model, null, commands);
    }

    private static final class EchoCommand implements ChatCommand {
        @Override
        public String name() {
            return "echo";
        }

        @Override
        public String description() {
            return "Repeat arguments";
        }

        @Override
        public void execute(final ChatCommandContext ctx) {
            ctx.model().append(ChatMessage.system(ctx.args()));
        }
    }
}
