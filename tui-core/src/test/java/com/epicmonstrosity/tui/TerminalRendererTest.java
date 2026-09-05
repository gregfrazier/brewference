package com.epicmonstrosity.tui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TerminalRendererTest {
    @Test
    void appendHeaderWritesExactlyTheReservedHeaderRows() {
        final StringBuilder frame = new StringBuilder();
        TerminalRenderer.appendHeader(frame, "a\nb\nc\n\n\nextra", 80);
        assertEquals(TerminalRenderer.HEADER_ROWS, countCrlf(frame.toString()));

        frame.setLength(0);
        TerminalRenderer.appendHeader(frame, "only-one", 80);
        assertEquals(TerminalRenderer.HEADER_ROWS, countCrlf(frame.toString()));
    }

    private static int countCrlf(final String text) {
        int count = 0;
        for (int i = 0; i < text.length() - 1; i++) {
            if (text.charAt(i) == '\r' && text.charAt(i + 1) == '\n') {
                count++;
            }
        }
        return count;
    }
}
