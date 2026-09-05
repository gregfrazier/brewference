package com.epicmonstrosity.tui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ViewContextTest {
    @Test
    void clampKeepsValueInsideRange() {
        assertEquals(0, ViewContext.clamp(-3, 10));
        assertEquals(10, ViewContext.clamp(12, 10));
        assertEquals(4, ViewContext.clamp(4, 10));
    }

    @Test
    void scrollOffsetKeepsCursorVisible() {
        assertEquals(0, ViewContext.scrollOffset(2, 0, 5));
        assertEquals(3, ViewContext.scrollOffset(3, 5, 5));
        assertEquals(1, ViewContext.scrollOffset(5, 0, 5));
        assertEquals(0, ViewContext.scrollOffset(0, 0, 0));
    }
}
