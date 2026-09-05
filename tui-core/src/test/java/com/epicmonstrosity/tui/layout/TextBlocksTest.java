package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.ansi.Ansi;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TextBlocksTest {
    @Test
    void visibleLengthUsesDisplayColumnsAndIgnoresAnsi() {
        assertEquals(4, TextBlocks.visibleLength(Ansi.GREEN + "你好" + Ansi.RESET));
        assertEquals(4, TextBlocks.visibleLength("a👋b"));
    }

    @Test
    void truncateKeepsAnsiAndDoesNotSplitAWideGlyph() {
        final String raw = Ansi.CYAN + "你好世界" + Ansi.RESET;
        final String cut = TextBlocks.truncateVisible(raw, 4);
        assertTrue(cut.contains(Ansi.CYAN));
        assertEquals("你好", TextBlocks.stripAnsi(cut));
        assertEquals(4, TextBlocks.visibleLength(cut));
    }

    @Test
    void fitPadsToDisplayWidth() {
        assertEquals("你 ", TextBlocks.fit("你好", 3));
        assertEquals(5, TextBlocks.visibleLength(TextBlocks.fit("hi", 5)));
    }
}
