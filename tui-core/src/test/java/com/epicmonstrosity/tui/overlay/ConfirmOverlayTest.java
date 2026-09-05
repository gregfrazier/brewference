package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Pasted;
import com.epicmonstrosity.tui.ScreenManager;
import com.epicmonstrosity.tui.TerminalRect;
import com.epicmonstrosity.tui.ViewContext;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfirmOverlayTest {
    @Test
    void yesAndEnterConfirmThenPop() {
        ScreenManager host = host();
        List<Boolean> results = new ArrayList<>();
        host.pushOverlay(ConfirmOverlay.yesNo("Delete file?", results::add));
        host.forwardEvent(new KeyPressed("y"));
        assertEquals(List.of(true), results);
        assertNull(host.peekOverlay());

        host.pushOverlay(ConfirmOverlay.yesNo("Again?", results::add));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of(true, true), results);
    }

    @Test
    void nAndEscCancelThenPop() {
        ScreenManager host = host();
        List<Boolean> results = new ArrayList<>();
        host.pushOverlay(ConfirmOverlay.yesNo("Sure?", results::add));
        host.forwardEvent(new KeyPressed("n"));
        assertEquals(List.of(false), results);

        host.pushOverlay(ConfirmOverlay.yesNo("Sure?", results::add));
        host.forwardEvent(new KeyPressed("ESC"));
        assertEquals(List.of(false, false), results);
        assertNull(host.peekOverlay());
    }

    @Test
    void typeToConfirmRequiresExactPhrase() {
        ScreenManager host = host();
        List<Boolean> results = new ArrayList<>();
        ConfirmOverlay overlay = ConfirmOverlay.typeToConfirm("Type DELETE", "DELETE", results::add);
        host.pushOverlay(overlay);

        host.forwardEvent(new KeyPressed("n"));
        host.forwardEvent(new KeyPressed("o"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of(), results);
        assertTrue(overlay.renderBody(ctx(), "").contains("DELETE"));

        host.forwardEvent(new KeyPressed("BACKSPACE"));
        host.forwardEvent(new KeyPressed("BACKSPACE"));
        host.forwardEvent(new KeyPressed("D"));
        host.forwardEvent(new KeyPressed("E"));
        host.forwardEvent(new KeyPressed("L"));
        host.forwardEvent(new KeyPressed("E"));
        host.forwardEvent(new KeyPressed("T"));
        host.forwardEvent(new KeyPressed("E"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of(true), results);
        assertNull(host.peekOverlay());
    }

    @Test
    void typeToConfirmAcceptsASinglePaste() {
        ScreenManager host = host();
        List<Boolean> results = new ArrayList<>();
        host.pushOverlay(ConfirmOverlay.typeToConfirm("Type DELETE", "DELETE", results::add));
        host.forwardEvent(new Pasted("DELETE"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of(true), results);
        assertNull(host.peekOverlay());
    }

    private static ScreenManager host() {
        return new ScreenManager(new TerminalRect() {
            @Override
            public int getWidth() {
                return 80;
            }

            @Override
            public int getHeight() {
                return 24;
            }
        });
    }

    private static ViewContext ctx() {
        return new ViewContext(24, 80, 15);
    }
}
