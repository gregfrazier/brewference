package com.epicmonstrosity.brewference.cli.screen.chat;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatModelTest {
    @Test
    void submitAppendsUserMessageAndFollowsBottom() {
        final ChatModel model = new ChatModel();
        model.append(ChatMessage.system("hello"));
        model.submitUser("hi");

        assertEquals(2, model.messages().size());
        assertEquals(ChatMessage.USER, model.messages().get(1).role());
        assertTrue(model.stickToBottom());
        assertEquals("hi", model.visibleWindow(1).getFirst().text());
    }

    @Test
    void scrollingUpDisablesStickToBottom() {
        final ChatModel model = new ChatModel();
        for (int i = 0; i < 10; i++) {
            model.append(ChatMessage.assistant("m" + i));
        }
        model.scrollBy(-3, 3);

        assertFalse(model.stickToBottom());
        assertEquals("m4", model.visibleWindow(3).getFirst().text());

        model.append(ChatMessage.assistant("new"));
        assertEquals("m4", model.visibleWindow(3).getFirst().text());
        assertFalse(model.stickToBottom());
    }

    @Test
    void scrollingBackToEndResticks() {
        final ChatModel model = new ChatModel();
        for (int i = 0; i < 8; i++) {
            model.append(ChatMessage.user("u" + i));
        }
        model.pageUp(3);
        assertFalse(model.stickToBottom());

        model.pageDown(3);
        model.pageDown(3);
        assertTrue(model.stickToBottom());
        model.append(ChatMessage.assistant("tail"));
        assertEquals("tail", model.visibleWindow(3).getLast().text());
    }

    @Test
    void streamingDeltasAppendToInProgressAssistant() {
        final ChatModel model = new ChatModel();
        model.startStreaming(ChatMessage.ASSISTANT);
        model.appendDelta("Hel");
        model.appendDelta("lo");
        assertTrue(model.streaming());
        assertEquals("Hello", model.messages().getFirst().text());
        assertTrue(model.messages().getFirst().streaming());

        model.finishStreaming();
        assertFalse(model.streaming());
        assertEquals("Hello", model.messages().getFirst().text());
        assertFalse(model.messages().getFirst().streaming());
    }

    @Test
    void pasteAndNewlineEditTheInputBuffer() {
        final ChatModel model = new ChatModel();
        model.paste("hello\r\nthere");
        model.newline();
        model.type("!");
        assertEquals("hello\nthere\n!", model.inputBuffer());
    }

    @Test
    void clearEmptiesMessagesInputAndScrollState() {
        final ChatModel model = new ChatModel();
        model.append(ChatMessage.user("one"));
        model.append(ChatMessage.assistant("two"));
        model.type("draft");
        model.pageUp(3);

        model.clear();

        assertTrue(model.messages().isEmpty());
        assertEquals("", model.inputBuffer());
        assertEquals(0, model.scrollOffset());
        assertTrue(model.stickToBottom());
    }

    @Test
    void visibleLinesUseARoleGutterAndWrapTheBody() {
        final ChatModel model = new ChatModel();
        model.append(ChatMessage.user("abcdefghij"));
        final List<String> lines = model.visibleLines(20, 10);
        Assertions.assertTrue(lines.getFirst().startsWith("user"));
        Assertions.assertTrue(lines.getFirst().contains("│"));
        Assertions.assertTrue(lines.size() >= 2);
        Assertions.assertTrue(lines.get(1).startsWith(" "));
        Assertions.assertTrue(lines.get(1).contains("│"));
        Assertions.assertTrue(lines.stream().anyMatch(line -> line.contains("abcd") || line.contains("efgh")));
    }

    @Test
    void emptyStreamingMessageShowsChurnUntilTheFirstToken() {
        final ChatModel model = new ChatModel();
        model.startGenerating();
        assertTrue(model.generating());
        assertTrue(model.streaming());

        final String first = model.churnText(0L);
        final String next = model.churnText(ChatModel.CHURN_FRAME_NANOS);
        Assertions.assertTrue(first.contains("generating"));
        Assertions.assertTrue(next.contains("generating"));
        Assertions.assertNotEquals(first, next);

        final List<String> lines = model.visibleLines(40, 5, 0L);
        Assertions.assertTrue(lines.getFirst().contains("assistant"));
        Assertions.assertTrue(lines.getFirst().contains("generating"));

        model.appendDelta("Hi");
        assertFalse(model.generating());
        assertTrue(model.visibleLines(40, 5).getFirst().contains("Hi"));
        assertFalse(model.visibleLines(40, 5).getFirst().contains("generating"));
    }
}
