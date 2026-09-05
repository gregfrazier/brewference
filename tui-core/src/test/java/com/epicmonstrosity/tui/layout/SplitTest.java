package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.ansi.Ansi;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SplitTest {
    @Test
    void defaultPercentIsFiftyFifty() {
        final SplitSizes sizes = Split.sizes(80);
        assertEquals(39, sizes.first());
        assertEquals(40, sizes.second());
        assertEquals(1, sizes.divider());
        assertEquals(80, sizes.total());
    }

    @Test
    void percentSplitsTheRemainderAfterTheDivider() {
        final SplitSizes sizes = Split.sizes(80, 40);
        assertEquals(31, sizes.first());
        assertEquals(48, sizes.second());
        assertEquals(1, sizes.divider());
    }

    @Test
    void percentIsClampedSoNeitherPaneIsZeroWhenThereIsRoom() {
        assertEquals(new SplitSizes(1, 78, 1), Split.sizes(80, 0));
        assertEquals(new SplitSizes(78, 1, 1), Split.sizes(80, 100));
    }

    @Test
    void tinyTotalsDropTheDivider() {
        assertEquals(new SplitSizes(0, 0, 0), Split.sizes(0));
        assertEquals(new SplitSizes(1, 0, 0), Split.sizes(1));
    }

    @Test
    void verticalStitchesTwoBlocksAtFiftyPercent() {
        final String out = visible(Split.vertical("AA\nBB", "CC", 7, 2));
        assertEquals("AA │CC ", out.split("\n")[0]);
        assertEquals("BB │   ", out.split("\n")[1]);
    }

    @Test
    void verticalUsesCustomPercent() {
        final SplitSizes sizes = Split.sizes(10, 30);
        assertEquals(2, sizes.first());
        assertEquals(7, sizes.second());
        final String first = visible(Split.vertical("L", "RIGHT", 10, 1, 30));
        assertEquals("L │RIGHT  ", first);
    }

    @Test
    void horizontalStitchesWithARuleBetweenPanes() {
        final String out = visible(Split.horizontal("TOP", "BOT", 5, 3));
        final String[] lines = out.split("\n", -1);
        assertEquals(3, lines.length);
        assertEquals("TOP  ", lines[0]);
        assertEquals("─────", lines[1]);
        assertEquals("BOT  ", lines[2]);
    }

    @Test
    void ansiCodesDoNotCountTowardPaneWidth() {
        final String left = Ansi.GREEN + "AB" + Ansi.RESET;
        final String out = Split.vertical(left, "CD", 7, 1);
        assertTrue(out.contains(Ansi.GREEN));
        assertEquals("AB │CD ", visible(out));
    }

    private static String visible(final String text) {
        return TextBlocks.stripAnsi(text);
    }
}
