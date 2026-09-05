package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.ScreenManager;
import com.epicmonstrosity.tui.TerminalRect;
import com.epicmonstrosity.tui.ViewContext;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ProgressOverlayTest {
    @Test
    void determinateBarShowsCountsAndCompletePopsOnDrain() {
        ScreenManager host = new ScreenManager(fixed());
        ProgressOverlay overlay = ProgressOverlay.determinate("Scan", "Working", 0, 10);
        host.pushOverlay(overlay);
        overlay.setProgress(4, 10);
        overlay.setMessage("Halfway");

        String body = overlay.renderBody(new ViewContext(24, 80, 15), "under");
        assertTrue(body.contains("Halfway"));
        assertTrue(body.contains("4/10"));

        overlay.complete();
        assertTrue(host.peekOverlay() == overlay);
        host.drainPostedEvents();
        assertNull(host.peekOverlay());
    }

    @Test
    void escapeCancelsWhenHandlerProvided() {
        ScreenManager host = new ScreenManager(fixed());
        AtomicBoolean cancelled = new AtomicBoolean();
        ProgressOverlay overlay = ProgressOverlay.indeterminate("Wait", "Please wait")
                .onCancel(() -> cancelled.set(true));
        host.pushOverlay(overlay);
        host.forwardEvent(new KeyPressed("ESC"));
        assertTrue(cancelled.get());
        assertNull(host.peekOverlay());
    }

    @Test
    void escapeDoesNothingWithoutCancelHandler() {
        ScreenManager host = new ScreenManager(fixed());
        ProgressOverlay overlay = ProgressOverlay.indeterminate("Wait", "Please wait");
        host.pushOverlay(overlay);
        host.forwardEvent(new KeyPressed("ESC"));
        assertFalse(overlay.renderBody(new ViewContext(24, 80, 15), "").isBlank());
        assertTrue(host.peekOverlay() == overlay);
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
