package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.overlay.HelpOverlay;
import com.epicmonstrosity.tui.screen.KeyBinding;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ScreenManagerTest {
    @Test
    void pushPopRestoresPreviousScreenWithoutFabricatingARoot() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen first = new FakeScreen("first");
        FakeScreen second = new FakeScreen("second");

        manager.push(first);
        manager.push(second);

        assertSame(second, manager.peek());
        assertEquals(List.of("enter"), second.events);
        assertEquals(List.of("enter"), first.events);

        manager.pop();
        assertSame(first, manager.peek());
        assertEquals(List.of("enter", "exit"), second.events);
        assertEquals(List.of("enter", "enter"), first.events);

        manager.pop();
        assertSame(first, manager.peek());
        assertEquals(List.of("enter", "exit"), second.events);
    }

    @Test
    void emptyStackPeekIsNullUntilAScreenIsPushed() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        assertNull(manager.peek());
    }

    @Test
    void overlayReceivesKeysWithoutExitingTheScreen() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen screen = new FakeScreen("first");
        FakeOverlay overlay = new FakeOverlay();
        manager.push(screen);
        manager.pushOverlay(overlay);

        assertSame(screen, manager.peek());
        assertSame(overlay, manager.peekOverlay());
        assertEquals(List.of("enter"), screen.events);

        manager.forwardEvent(new KeyPressed("x"));
        assertEquals(List.of("event:x"), overlay.events);
        assertEquals(List.of("enter"), screen.events);

        manager.forwardEvent(new KeyPressed("ESC"));
        assertNull(manager.peekOverlay());
        assertSame(screen, manager.peek());
        assertEquals(List.of("enter"), screen.events);
    }

    @Test
    void helpOverlayListsBindingsAndEscCloses() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen screen = new FakeScreen("first");
        manager.push(screen);
        manager.pushOverlay(HelpOverlay.from(screen));

        String body = manager.peekOverlay().renderBody(new ViewContext(24, 80, 15), "underlying");
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("[x] example"));
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("Help"));

        manager.forwardEvent(new KeyPressed("ESC"));
        assertNull(manager.peekOverlay());
        assertEquals(List.of("enter"), screen.events);
    }

    @Test
    void eventsGoToTheCurrentScreen() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen first = new FakeScreen("first");
        FakeScreen second = new FakeScreen("second");
        manager.push(first);
        manager.push(second);

        manager.forwardEvent(new KeyPressed("x"));
        assertEquals(List.of("enter", "event:x"), second.events);
        assertEquals(List.of("enter"), first.events);

        manager.bubbleEvent(new KeyPressed("b"));
        assertEquals(List.of("enter", "event:b"), first.events);
    }

    @Test
    void replaceExitsCurrentAndEntersNewWithoutReenteringPrevious() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen first = new FakeScreen("first");
        FakeScreen second = new FakeScreen("second");
        FakeScreen third = new FakeScreen("third");

        manager.push(first);
        manager.push(second);
        manager.replace(third);

        assertSame(third, manager.peek());
        assertEquals(List.of("enter", "exit"), second.events);
        assertEquals(List.of("enter"), first.events);
        assertEquals(List.of("enter"), third.events);

        manager.pop();
        assertSame(first, manager.peek());
        assertEquals(List.of("enter", "enter"), first.events);
    }

    @Test
    void replaceOnRootDoesNotFabricateAFallback() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen first = new FakeScreen("first");
        FakeScreen second = new FakeScreen("second");
        manager.push(first);
        manager.replace(second);

        assertSame(second, manager.peek());
        assertEquals(List.of("enter", "exit"), first.events);
        manager.pop();
        assertSame(second, manager.peek());
    }

    @Test
    void replaceClearsOverlaysWithoutExitingTheReplacedScreenTwice() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen first = new FakeScreen("first");
        FakeScreen second = new FakeScreen("second");
        manager.push(first);
        manager.pushOverlay(new FakeOverlay());
        manager.replace(second);

        assertNull(manager.peekOverlay());
        assertSame(second, manager.peek());
        assertEquals(List.of("enter", "exit"), first.events);
    }

    @Test
    void postQueuesUntilDrainAndRoutesToOverlayFirst() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen screen = new FakeScreen("first");
        FakeOverlay overlay = new FakeOverlay();
        manager.push(screen);
        manager.post(new KeyPressed("x"));
        assertEquals(List.of("enter"), screen.events);

        assertEquals(1, manager.drainPostedEvents());
        assertEquals(List.of("enter", "event:x"), screen.events);

        manager.pushOverlay(overlay);
        manager.post(new KeyPressed("y"));
        manager.drainPostedEvents();
        assertEquals(List.of("event:y"), overlay.events);
        assertEquals(List.of("enter", "event:x"), screen.events);
    }

    @Test
    void postInvalidatesUntilTheInboxIsDrained() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen screen = new FakeScreen("first");
        manager.push(screen);
        assertFalse(manager.isCurrentScreenInvalidated());

        manager.post(new Tick());
        assertTrue(manager.isCurrentScreenInvalidated());
        assertEquals(List.of("enter"), screen.events);

        assertEquals(1, manager.drainPostedEvents());
        assertEquals(List.of("enter", "event"), screen.events);
        assertFalse(manager.isCurrentScreenInvalidated());
    }

    @Test
    void postIgnoresNullAndDoesNotInvalidate() {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        manager.push(new FakeScreen("first"));
        manager.post(null);
        assertFalse(manager.isCurrentScreenInvalidated());
        assertEquals(0, manager.drainPostedEvents());
    }

    @Test
    void workersCanPostWithoutCallingForwardEvent() throws Exception {
        ScreenManager manager = new ScreenManager(fixedTerminal());
        FakeScreen screen = new FakeScreen("first");
        manager.push(screen);

        final int workers = 8;
        final int perWorker = 25;
        final CyclicBarrier start = new CyclicBarrier(workers);
        final CountDownLatch done = new CountDownLatch(workers);
        final AtomicInteger failures = new AtomicInteger();

        for (int w = 0; w < workers; w++) {
            final int worker = w;
            Thread.ofVirtual().start(() -> {
                try {
                    start.await(2, TimeUnit.SECONDS);
                    for (int i = 0; i < perWorker; i++) {
                        manager.post(new KeyPressed(worker + ":" + i));
                    }
                } catch (Exception e) {
                    failures.incrementAndGet();
                } finally {
                    done.countDown();
                }
            });
        }

        assertTrue(done.await(5, TimeUnit.SECONDS));
        assertEquals(0, failures.get());
        assertTrue(manager.isCurrentScreenInvalidated());
        assertEquals(workers * perWorker, manager.drainPostedEvents());
        assertEquals(1 + workers * perWorker, screen.events.size());
        assertFalse(manager.isCurrentScreenInvalidated());
    }

    private static TerminalRect fixedTerminal() {
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

    private static final class FakeScreen implements Screen {
        private final String name;
        private final List<String> events = new ArrayList<>();

        private FakeScreen(String name) {
            this.name = name;
        }

        @Override
        public String[] render(ViewContext ctx) {
            return new String[]{name, "", ""};
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
            events.add("exit");
        }

        @Override
        public void onEvent(UiMessage msg) {
            if (msg instanceof KeyPressed(String key)) {
                events.add("event:" + key);
            } else {
                events.add("event");
            }
        }

        @Override
        public boolean isInvalidated() {
            return false;
        }

        @Override
        public List<KeyBinding> keyBindings() {
            return List.of(KeyBinding.primary("x", "x", "example"));
        }
    }

    private static final class FakeOverlay implements Overlay {
        private UiHost host;
        private final List<String> events = new ArrayList<>();

        @Override
        public void onOpen(UiHost host) {
            this.host = host;
        }

        @Override
        public String renderBody(ViewContext ctx, String underlyingBody) {
            return "overlay";
        }

        @Override
        public void onEvent(UiMessage msg) {
            if (msg instanceof KeyPressed(String key)) {
                events.add("event:" + key);
                if ("ESC".equals(key) && host != null) {
                    host.popOverlay();
                }
            }
        }

        @Override
        public boolean isInvalidated() {
            return false;
        }
    }
}
