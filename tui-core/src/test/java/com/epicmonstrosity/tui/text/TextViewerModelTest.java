package com.epicmonstrosity.tui.text;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TextViewerModelTest {
    @Test
    void wrapSplitsLongLinesToTerminalWidth() {
        TextViewerModel model = TextViewerModel.from("abcdefghij");
        assertEquals(List.of("abcde", "fghij"), model.wrap(5));
    }

    @Test
    void wrapUsesDisplayColumnsForWideGlyphs() {
        TextViewerModel model = TextViewerModel.from("你好世界");
        assertEquals(List.of("你好", "世界"), model.wrap(4));
    }

    @Test
    void pageDownAndEndReachLastWindow() {
        TextViewerModel model = TextViewerModel.fromLines(List.of("a", "b", "c", "d", "e", "f"));
        model.pageDown(10, 2);
        assertEquals(List.of("c", "d"), model.visibleWindow(10, 2));

        model.end(10, 2);
        assertEquals(List.of("e", "f"), model.visibleWindow(10, 2));
        assertEquals(4, model.scrollOffset());
    }

    @Test
    void homeReturnsToStart() {
        TextViewerModel model = TextViewerModel.from("one two three four");
        model.end(4, 2);
        model.home();
        assertEquals(0, model.scrollOffset());
        assertEquals(List.of("one ", "two "), model.visibleWindow(4, 2));
    }

    @Test
    void findNextAndPreviousMoveToWrappedMatches() {
        TextViewerModel model = TextViewerModel.fromLines(List.of("alpha", "beta", "alpha again", "gamma"));
        model.setQuery("alpha");
        assertTrue(model.findNext(20, 2));
        assertEquals(0, model.matchIndex());
        assertEquals(0, model.scrollOffset());

        assertTrue(model.findNext(20, 2));
        assertEquals(2, model.matchIndex());
        assertEquals(2, model.scrollOffset());

        assertTrue(model.findPrevious(20, 2));
        assertEquals(0, model.matchIndex());
    }

    @Test
    void missingQueryDoesNotMove() {
        TextViewerModel model = TextViewerModel.from("nothing here");
        model.setQuery("zzz");
        assertFalse(model.findNext(20, 5));
        assertEquals(-1, model.matchIndex());
        assertEquals(0, model.scrollOffset());
    }

    @Test
    void appendWhileFollowingPinsToLastWindow() {
        TextViewerModel model = TextViewerModel.fromLines(List.of("a", "b")).followTail();
        model.append("c\nd\ne\n");
        assertTrue(model.following());
        assertEquals(List.of("d", "e"), model.visibleWindow(10, 2));
    }

    @Test
    void scrollingUpPausesFollowSoNewLinesDoNotJump() {
        TextViewerModel model = TextViewerModel.fromLines(List.of("a", "b", "c", "d")).followTail();
        model.visibleWindow(10, 2);
        model.moveBy(-1, 10, 2);

        assertFalse(model.following());
        assertEquals(List.of("b", "c"), model.visibleWindow(10, 2));

        model.append("e\nf\n");
        assertFalse(model.following());
        assertEquals(List.of("b", "c"), model.visibleWindow(10, 2));
    }

    @Test
    void endResticksFollow() {
        TextViewerModel model = TextViewerModel.fromLines(List.of("a", "b", "c", "d", "e")).followTail();
        model.home();
        assertFalse(model.following());

        model.end(10, 2);
        assertTrue(model.following());
        model.append("tail\n");
        assertEquals(List.of("e", "tail"), model.visibleWindow(10, 2));
    }

    @Test
    void incompleteLineIsCompletedOnNextAppend() {
        TextViewerModel model = TextViewerModel.from("");
        model.append("hel");
        model.append("lo\nworld");
        assertEquals(List.of("hello", "world"), model.sourceLines());
    }

    @Test
    void pollReadsNewFileBytesAndReloadsOnTruncate() throws Exception {
        Path log = Files.createTempFile("tui-tail", ".log");
        try {
            Files.writeString(log, "one\n");
            TextViewerModel model = TextViewerModel.tail(log);
            assertEquals(List.of("one"), model.sourceLines());
            assertTrue(model.following());

            Files.writeString(log, "two\n", StandardOpenOption.APPEND);
            assertTrue(model.poll());
            assertEquals(List.of("one", "two"), model.sourceLines());
            assertFalse(model.poll());

            Files.writeString(log, "x\n");
            assertTrue(model.poll());
            assertEquals(List.of("x"), model.sourceLines());
        } finally {
            Files.deleteIfExists(log);
        }
    }
}
