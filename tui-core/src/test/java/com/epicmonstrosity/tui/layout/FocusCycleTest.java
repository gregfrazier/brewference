package com.epicmonstrosity.tui.layout;

import com.epicmonstrosity.tui.KeyPressed;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FocusCycleTest {
    @Test
    void startsOnTheFirstRegion() {
        final FocusCycle focus = new FocusCycle("list", "detail");
        assertTrue(focus.is("list"));
        assertEquals("list", focus.current());
        assertEquals(0, focus.index());
        assertEquals(2, focus.size());
    }

    @Test
    void tabAndShiftTabWrap() {
        final FocusCycle focus = new FocusCycle("list", "detail", "log");
        assertTrue(focus.handle(new KeyPressed("TAB")));
        assertTrue(focus.is("detail"));
        assertTrue(focus.handle(new KeyPressed("TAB")));
        assertTrue(focus.is("log"));
        assertTrue(focus.handle(new KeyPressed("TAB")));
        assertTrue(focus.is("list"));
        assertTrue(focus.handle(new KeyPressed("SHIFT_TAB")));
        assertTrue(focus.is("log"));
    }

    @Test
    void unusedKeysAreNotConsumed() {
        final FocusCycle focus = new FocusCycle("list", "detail");
        assertFalse(focus.handle(new KeyPressed("DOWN")));
        assertTrue(focus.is("list"));
    }

    @Test
    void focusJumpsByName() {
        final FocusCycle focus = new FocusCycle("list", "detail");
        focus.focus("detail");
        assertTrue(focus.is("detail"));
        focus.focus("missing");
        assertTrue(focus.is("detail"));
    }

    @Test
    void requiresAtLeastOneRegion() {
        assertThrows(IllegalArgumentException.class, FocusCycle::new);
    }
}
