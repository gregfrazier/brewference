package com.epicmonstrosity.tui.overlay;

import com.epicmonstrosity.tui.KeyPressed;
import com.epicmonstrosity.tui.Screen;
import com.epicmonstrosity.tui.ScreenManager;
import com.epicmonstrosity.tui.TerminalRect;
import com.epicmonstrosity.tui.UiHost;
import com.epicmonstrosity.tui.UiMessage;
import com.epicmonstrosity.tui.ViewContext;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToastTest {
    @Test
    void showToastDoesNotStealKeysOrPushAnOverlay() {
        ScreenManager manager = host();
        FakeScreen screen = new FakeScreen();
        manager.push(screen);
        manager.showToast(Toast.success("Saved"));

        assertSame(Toast.Kind.SUCCESS, manager.peekToast().kind());
        assertNull(manager.peekOverlay());

        manager.forwardEvent(new KeyPressed("x"));
        assertEquals(List.of("enter", "event:x"), screen.events);
        assertEquals("Saved", manager.peekToast().message());
    }

    @Test
    void expiredToastIsSweptAndForcesARedraw() throws Exception {
        ScreenManager manager = host();
        manager.showToast(new Toast("gone", Toast.Kind.INFO, 1));
        assertTrue(manager.peekToast() != null);

        Thread.sleep(5);
        assertTrue(manager.isCurrentScreenInvalidated());
        assertTrue(manager.sweepToast());
        assertNull(manager.peekToast());
        assertFalse(manager.sweepToast());
    }

    @Test
    void queuedToastsAreAllVisibleAtOnce() {
        ScreenManager manager = host();
        manager.showToast(Toast.info("first"));
        manager.showToast(Toast.success("second"));
        manager.showToast(Toast.warning("third"));

        assertEquals(List.of("first", "second", "third"), messages(manager.visibleToasts()));
        assertEquals("third", manager.peekToast().message());
        assertEquals(0, manager.pendingToastCount());
    }

    @Test
    void overflowToastDoesNotExpireUntilItBecomesVisible() throws Exception {
        ScreenManager manager = host();
        manager.showToast(new Toast("one", Toast.Kind.INFO, 1));
        manager.showToast(new Toast("two", Toast.Kind.INFO, 1));
        manager.showToast(new Toast("three", Toast.Kind.INFO, 1));
        manager.showToast(new Toast("four", Toast.Kind.SUCCESS, 1));

        assertEquals(List.of("one", "two", "three"), messages(manager.visibleToasts()));
        assertEquals(1, manager.pendingToastCount());

        Thread.sleep(5);
        assertTrue(manager.sweepToast());
        assertEquals(List.of("four"), messages(manager.visibleToasts()));
        assertEquals("four", manager.visibleToast().message());
    }

    @Test
    void queuedToastsPromoteAfterTheCurrentOneExpires() throws Exception {
        ScreenManager manager = host();
        manager.showToast(new Toast("first", Toast.Kind.INFO, 1));
        manager.showToast(Toast.success("second"));
        manager.showToast(Toast.warning("third"));

        assertEquals(List.of("first", "second", "third"), messages(manager.visibleToasts()));

        Thread.sleep(5);
        assertTrue(manager.sweepToast());
        assertEquals(List.of("second", "third"), messages(manager.visibleToasts()));
        assertEquals("third", manager.peekToast().message());

        manager.dismissToast();
        assertEquals("second", manager.peekToast().message());
        assertEquals(0, manager.pendingToastCount());
    }

    @Test
    void stickyStatusStaysUntilClearedAndDoesNotStealKeys() {
        ScreenManager manager = host();
        FakeScreen screen = new FakeScreen();
        manager.push(screen);

        manager.setStatus(Toast.info("Scanning…"));
        assertEquals("Scanning…", manager.peekStatus().message());
        assertEquals("Scanning…", manager.visibleToast().message());
        assertNull(manager.peekToast());

        manager.showToast(Toast.success("Copied"));
        assertEquals("Copied", manager.visibleToast().message());
        assertEquals("Scanning…", manager.peekStatus().message());

        manager.dismissToast();
        assertEquals("Scanning…", manager.visibleToast().message());

        manager.forwardEvent(new KeyPressed("x"));
        assertEquals(List.of("enter", "event:x"), screen.events);

        manager.clearStatus();
        assertNull(manager.peekStatus());
        assertNull(manager.visibleToast());
    }

    @Test
    void dismissClearsAStickyToast() {
        ScreenManager manager = host();
        manager.showToast(new Toast("stay", Toast.Kind.WARNING, 0));
        assertEquals("stay", manager.peekToast().message());
        manager.dismissToast();
        assertNull(manager.peekToast());
    }

    @Test
    void bannerPaintsABoxAboveTheFooterGap() {
        final String painted = ToastBanner.paint(
                "one\ntwo\nthree\nfour\nfive",
                List.of(Toast.info("Copied")),
                null,
                24,
                5
        );
        final String[] lines = painted.split("\n", -1);
        assertEquals(5, lines.length);
        assertEquals("one", lines[0]);
        assertTrue(lines[1].contains("┌"));
        assertTrue(lines[2].contains("Copied"));
        assertTrue(lines[3].contains("└"));
        assertEquals("five", lines[4]);
    }

    @Test
    void bannerStacksQueuedToastsWithoutClippingTheBottomBox() {
        final String painted = ToastBanner.paint(
                "keep\na\nb\nc\nd\ne\nf\ng\nh\ni",
                List.of(Toast.info("first"), Toast.success("second"), Toast.warning("third")),
                null,
                28,
                10
        );
        final String[] lines = painted.split("\n", -1);
        assertEquals(10, lines.length);
        assertTrue(painted.contains("first"));
        assertTrue(painted.contains("second"));
        assertTrue(painted.contains("third"));
        assertEquals("i", lines[9]);
        assertTrue(lines[8].contains("└"));
    }

    @Test
    void bannerShowsOverflowCountInTheTitle() {
        final String painted = ToastBanner.paint(
                "body",
                List.of(Toast.success("Saved")),
                null,
                24,
                4,
                2
        );
        assertTrue(painted.contains("+2"));
        assertTrue(painted.contains("Saved"));
        assertTrue(painted.contains("ok"));
    }

    private static List<String> messages(final List<Toast> toasts) {
        return toasts.stream().map(Toast::message).toList();
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

    private static final class FakeScreen implements Screen {
        private final List<String> events = new ArrayList<>();

        @Override
        public String[] render(ViewContext ctx) {
            return new String[]{"h", "b", "f"};
        }

        @Override
        public Screen handleInput(String key) {
            return this;
        }

        @Override
        public void onEnter(UiHost host) {
            events.add("enter");
        }

        @Override
        public void onExit() {
        }

        @Override
        public void onEvent(UiMessage msg) {
            if (msg instanceof KeyPressed(String key)) {
                events.add("event:" + key);
            }
        }

        @Override
        public boolean isInvalidated() {
            return false;
        }
    }
}
