package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.ScreenManager;
import com.epicmonstrosity.tui.TerminalRect;
import com.epicmonstrosity.tui.ViewContext;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChoiceOverlayTest {
    @Test
    void enterPicksTheHighlightedItem() {
        ScreenManager host = host();
        List<String> picked = new ArrayList<>();
        host.pushOverlay(ChoiceOverlay.of("Role", List.of("Admin", "User", "Guest"), picked::add));

        host.forwardEvent(new KeyPressed("DOWN"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of("User"), picked);
        assertNull(host.peekOverlay());
    }

    @Test
    void escCancelsWithoutACallback() {
        ScreenManager host = host();
        List<String> picked = new ArrayList<>();
        host.pushOverlay(ChoiceOverlay.of("Role", List.of("Admin", "User"), picked::add));
        host.forwardEvent(new KeyPressed("ESC"));
        assertEquals(List.of(), picked);
        assertNull(host.peekOverlay());
    }

    @Test
    void multiSelectTogglesWithSpaceAndSubmitsOnEnter() {
        ScreenManager host = host();
        List<List<String>> picked = new ArrayList<>();
        host.pushOverlay(ChoiceOverlay.multi("Tags", List.of("a", "b", "c"), String::valueOf, picked::add));

        host.forwardEvent(new KeyPressed("SPACE"));
        host.forwardEvent(new KeyPressed("DOWN"));
        host.forwardEvent(new KeyPressed("DOWN"));
        host.forwardEvent(new KeyPressed("SPACE"));
        host.forwardEvent(new KeyPressed("ENTER"));
        assertEquals(List.of(List.of("a", "c")), picked);
        assertNull(host.peekOverlay());
    }

    @Test
    void renderListsChoicesInAModal() {
        ChoiceOverlay<String> overlay = ChoiceOverlay.of("Role", List.of("Admin", "User"), ignored -> {});
        overlay.onOpen(host());
        final String body = overlay.renderBody(new ViewContext(24, 80, 15), "under");
        assertTrue(body.contains("Role"));
        assertTrue(body.contains("Admin"));
        assertTrue(body.contains("User"));
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
}
