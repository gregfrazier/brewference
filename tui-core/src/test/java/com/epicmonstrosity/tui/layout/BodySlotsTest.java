package com.epicmonstrosity.tui.layout;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BodySlotsTest {
    @Test
    void verticalSlotsExposePaneSizesForWrapping() {
        final BodySlots slots = BodySlots.vertical(80, 20, 40);
        assertEquals(31, slots.firstWidth());
        assertEquals(20, slots.firstHeight());
        assertEquals(48, slots.secondWidth());
        assertEquals(20, slots.secondHeight());
    }

    @Test
    void horizontalSlotsExposePaneSizes() {
        final BodySlots slots = BodySlots.horizontal(40, 10);
        assertEquals(40, slots.firstWidth());
        assertEquals(4, slots.firstHeight());
        assertEquals(40, slots.secondWidth());
        assertEquals(5, slots.secondHeight());
    }

    @Test
    void namedPutsRenderTheSameAsFluentSetters() {
        final String fluent = BodySlots.vertical(7, 1)
                .left("AA")
                .right("CC")
                .render();
        final String named = BodySlots.vertical(7, 1)
                .put("left", "AA")
                .put("right", "CC")
                .render();
        assertEquals(TextBlocks.stripAnsi(fluent), TextBlocks.stripAnsi(named));
        assertEquals("AA │CC ", TextBlocks.stripAnsi(fluent));
    }

    @Test
    void unknownSlotNameFailsFast() {
        assertThrows(IllegalArgumentException.class, () -> BodySlots.vertical(10, 4).put("middle", "x"));
    }
}
