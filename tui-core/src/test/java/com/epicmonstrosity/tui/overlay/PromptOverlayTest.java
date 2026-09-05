package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.ScreenManager;
import com.epicmonstrosity.tui.TerminalRect;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class PromptOverlayTest {
    @Test
    void enterSubmitsEditedValue() {
        ScreenManager host = new ScreenManager(fixed());
        List<String> submitted = new ArrayList<>();
        host.pushOverlay(PromptOverlay.ask("Name", "Ada", submitted::add));
        host.forwardEvent(new KeyPressed("BACKSPACE"));
        host.forwardEvent(new KeyPressed("BACKSPACE"));
        host.forwardEvent(new KeyPressed("BACKSPACE"));
        host.forwardEvent(new KeyPressed("B"));
        host.forwardEvent(new KeyPressed("o"));
        host.forwardEvent(new KeyPressed("b"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of("Bob"), submitted);
        assertNull(host.peekOverlay());
    }

    @Test
    void escapeCancelsWithoutSubmit() {
        ScreenManager host = new ScreenManager(fixed());
        List<String> submitted = new ArrayList<>();
        host.pushOverlay(PromptOverlay.ask("Name", "Ada", submitted::add));
        host.forwardEvent(new KeyPressed("x"));
        host.forwardEvent(new KeyPressed("ESC"));
        assertEquals(List.of(), submitted);
        assertNull(host.peekOverlay());
    }

    private static TerminalRect fixed() {
        return new TerminalRect() {
            @Override
            public int getWidth() {
                return 80;
            }

            @Override
            public int getHeight() {
                return 24;
            }
        };
    }
}
