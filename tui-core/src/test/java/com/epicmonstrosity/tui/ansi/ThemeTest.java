package com.epicmonstrosity.tui.ansi;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class ThemeTest {
    @AfterEach
    void restoreDefault() {
        Theme.setCurrent(Theme.DEFAULT);
    }

    @Test
    void defaultThemeUsesColorSequences() {
        assertFalse(Theme.DEFAULT.error().isBlank());
        assertEquals(Ansi.RESET, Theme.DEFAULT.reset());
        assertEquals(Ansi.RED, Theme.DEFAULT.error());
        assertEquals(Ansi.GREEN, Theme.DEFAULT.success());
    }

    @Test
    void monoThemeDropsForegroundColors() {
        assertEquals(Ansi.RESET, Theme.MONO.reset());
        assertEquals("", Theme.MONO.accent());
        assertEquals("", Theme.MONO.error());
        assertEquals(Ansi.BOLD, Theme.MONO.bold());
        assertEquals(Ansi.DIM, Theme.MONO.dim());
        assertEquals(Ansi.BG_SEL, Theme.MONO.selection());
    }

    @Test
    void setCurrentSwapsTheProcessTheme() {
        assertSame(Theme.DEFAULT, Theme.current());
        Theme.setCurrent(Theme.MONO);
        assertSame(Theme.MONO, Theme.current());
        Theme.setCurrent(null);
        assertSame(Theme.DEFAULT, Theme.current());
    }

    @Test
    void paintWrapsTextAndResets() {
        final String painted = Theme.DEFAULT.paint(Theme.DEFAULT.error(), "boom");
        assertTrueStartsWith(painted, Theme.DEFAULT.error());
        assertTrue(painted.endsWith(Theme.DEFAULT.reset()));
        assertNotEquals("boom", painted);
    }

    private static void assertTrueStartsWith(final String value, final String prefix) {
        org.junit.jupiter.api.Assertions.assertTrue(value.startsWith(prefix), value);
    }

    private static void assertTrue(final boolean value) {
        org.junit.jupiter.api.Assertions.assertTrue(value);
    }
}
