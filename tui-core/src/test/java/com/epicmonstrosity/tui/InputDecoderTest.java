package com.epicmonstrosity.tui;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InputDecoderTest {
    @Test
    void bracketedPasteIsASingleEvent() throws IOException {
        final String blob = "hello\r\nthere\n" + "x".repeat(400);
        final List<UiMessage> events = drain(bytes(
                27, '[', '2', '0', '0', '~',
                blob,
                27, '[', '2', '0', '1', '~'
        ));

        assertEquals(1, events.size());
        final Pasted pasted = assertInstanceOf(Pasted.class, events.getFirst());
        assertEquals("hello\nthere\n" + "x".repeat(400), pasted.text());
    }

    @Test
    void unbracketedBurstIsASinglePaste() throws IOException {
        final String blob = "paste me\r\nplease";
        final List<UiMessage> events = drain(bytes(blob));

        assertEquals(1, events.size());
        final Pasted pasted = assertInstanceOf(Pasted.class, events.getFirst());
        assertEquals("paste me\nplease", pasted.text());
    }

    @Test
    void singleTypedCharacterStaysAKey() throws IOException {
        final List<UiMessage> events = drain(bytes("a"));
        assertEquals(1, events.size());
        assertEquals(new KeyPressed("a"), events.getFirst());
    }

    @Test
    void shiftTabIsASingleKey() throws IOException {
        assertEquals(new KeyPressed("SHIFT_TAB"), drain(new int[]{27, '[', 'Z'}).getFirst());
        assertEquals(new KeyPressed("TAB"), drain(new int[]{9}).getFirst());
    }

    @Test
    void loneEnterIsNotAPaste() throws IOException {
        assertEquals(new KeyPressed("ENTER"), drain(new int[]{13}).getFirst());
        assertEquals(new KeyPressed("CTRL_J"), drain(new int[]{10}).getFirst());
    }

    @Test
    void arrowsSurviveBesideAPaste() throws IOException {
        final List<UiMessage> events = drain(concat(
                new int[]{27, '[', 'A'},
                bytes("hello"),
                new int[]{27, '[', 'B'}
        ));
        assertEquals(3, events.size());
        assertEquals(new KeyPressed("UP"), events.get(0));
        assertEquals(new Pasted("hello"), events.get(1));
        assertEquals(new KeyPressed("DOWN"), events.get(2));
    }

    @Test
    void emptyBracketedPasteIsEmptyText() throws IOException {
        final List<UiMessage> events = drain(new int[]{27, '[', '2', '0', '0', '~', 27, '[', '2', '0', '1', '~'});
        assertEquals(1, events.size());
        assertEquals(new Pasted(""), events.getFirst());
    }

    @Test
    void eofWithoutBytesIsNull() throws IOException {
        assertNull(new InputDecoder(timeout -> timeout < 0
                ? InputDecoder.EOF
                : InputDecoder.EXPIRED).readEvent(-1, () -> {}));
    }

    @Test
    void idleCallbackRunsOnTimeoutThenReadsKey() throws IOException {
        final int[] calls = {0};
        final int[] step = {0};
        final InputDecoder decoder = new InputDecoder(timeout -> {
            if (step[0] == 0 && timeout >= 0) {
                step[0] = 1;
                return InputDecoder.EXPIRED;
            }
            if (step[0] == 1) {
                step[0] = 2;
                return 'z';
            }
            return timeout < 0 ? InputDecoder.EOF : InputDecoder.EXPIRED;
        });
        final UiMessage event = decoder.readEvent(100, () -> calls[0]++);
        assertEquals(new KeyPressed("z"), event);
        assertTrue(calls[0] >= 1);
    }

    private static List<UiMessage> drain(final int[] data) throws IOException {
        final ScriptedSource source = new ScriptedSource(data);
        final InputDecoder decoder = new InputDecoder(source);
        final List<UiMessage> events = new ArrayList<>();
        while (true) {
            final UiMessage event = decoder.readEvent(-1, () -> {});
            if (event == null) {
                return events;
            }
            events.add(event);
        }
    }

    private static int[] bytes(final String text) {
        final int[] out = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    private static int[] bytes(final int a, final int b, final int c, final int d, final int e, final int f,
                               final String body,
                               final int g, final int h, final int i, final int j, final int k, final int l) {
        return concat(new int[]{a, b, c, d, e, f}, bytes(body), new int[]{g, h, i, j, k, l});
    }

    private static int[] concat(final int[]... parts) {
        int size = 0;
        for (int[] part : parts) {
            size += part.length;
        }
        final int[] out = new int[size];
        int at = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, at, part.length);
            at += part.length;
        }
        return out;
    }

    private static final class ScriptedSource implements InputDecoder.Source {
        private final int[] data;
        private int index;

        private ScriptedSource(final int[] data) {
            this.data = data;
        }

        @Override
        public int read(final int timeoutMs) {
            if (index >= data.length) {
                return timeoutMs < 0 ? InputDecoder.EOF : InputDecoder.EXPIRED;
            }
            return data[index++];
        }
    }
}
