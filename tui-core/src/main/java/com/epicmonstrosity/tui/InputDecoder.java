package com.epicmonstrosity.tui;

import java.io.IOException;

/**
 * Turns a raw terminal byte stream into {@link KeyPressed} and {@link Pasted} events.
 * Bracketed paste ({@code ESC[200~} … {@code ESC[201~}) is one event; a burst of
 * printable characters without those markers is coalesced so a clipboard dump
 * is not delivered as hundreds of keys.
 */
public final class InputDecoder {
    public static final int EOF = -1;
    public static final int EXPIRED = -2;

    static final int ESC_TIMEOUT_MS = 50;
    static final int COALESCE_TIMEOUT_MS = 5;

    private static final int NONE = Integer.MIN_VALUE;
    private static final int ESC = 27;
    private static final int CSI_BRACKET = '[';
    private static final int CSI_O = 79;

    @FunctionalInterface
    public interface Source {
        /**
         * Read a single byte from the terminal with a timeout.
         *
         * @param timeoutMs The timeout in milliseconds to wait for input
         * @return The byte read, or -1 if EOF, -2 if timeout expired
         * @throws IOException If an I/O error occurs
         */
        int read(int timeoutMs) throws IOException;
    }

    private final Source source;
    private int unread = NONE;

    public InputDecoder(final Source source) {
        this.source = source;
    }

    /**
     * Block until the next event. {@code onIdle} runs whenever {@code idleTimeoutMs}
     * expires with no input (used to redraw on invalidate). {@code null} means EOF.
     *
     * @param idleTimeoutMs The timeout in milliseconds to wait for input
     * @param onIdle Callback to run when timeout expires with no input
     * @return The next UI event, or null for EOF
     * @throws IOException If an I/O error occurs
     */
    public UiMessage readEvent(final int idleTimeoutMs, final Runnable onIdle) throws IOException {
        final Runnable idle = onIdle == null ? () -> {} : onIdle;
        while (true) {
            final int c = next(idleTimeoutMs);
            if (c == EOF) {
                return null;
            }
            if (c == EXPIRED) {
                idle.run();
                continue;
            }
            return decode(c);
        }
    }

    /**
     * Decode the first byte into a UI event.
     *
     * @param first The first byte read from the terminal
     * @return The decoded UI event
     * @throws IOException If an I/O error occurs during decode
     */
    private UiMessage decode(final int first) throws IOException {
        if (first == ESC) {
            return decodeEscape();
        }
        if (isPasteChar(first)) {
            return decodePrintableOrPaste(first);
        }
        return new KeyPressed(controlKey(first));
    }

    /**
     * Decode printable characters or a bracketed paste sequence.
     *
     * @param first The first printable character
     * @return A KeyPressed for single chars, or Pasted for multi-char pastes
     * @throws IOException If an I/O error occurs
     */
    private UiMessage decodePrintableOrPaste(final int first) throws IOException {
        final StringBuilder buf = new StringBuilder();
        appendPasteChar(buf, first);
        while (true) {
            final int c = next(COALESCE_TIMEOUT_MS);
            if (c == EOF || c == EXPIRED) {
                break;
            }
            if (!isPasteChar(c)) {
                unread(c);
                break;
            }
            appendPasteChar(buf, c);
        }
        if (buf.length() == 1) {
            if (first == 9) {
                return new KeyPressed("TAB");
            }
            if (first == 10) {
                return new KeyPressed("CTRL_J");
            }
            if (first == 13) {
                return new KeyPressed("ENTER");
            }
            return new KeyPressed(buf.toString());
        }
        return new Pasted(buf.toString());
    }

    /**
     * Decode an escape sequence (ESC byte followed by additional characters).
     *
     * @return The decoded UI event for the escape sequence
     * @throws IOException If an I/O error occurs
     */
    private UiMessage decodeEscape() throws IOException {
        final int next = next(ESC_TIMEOUT_MS);
        if (next == EOF || next == EXPIRED || next == ESC) {
            return new KeyPressed("ESC");
        }

        if (next == CSI_BRACKET || next == CSI_O) {
            final int code = next(-1);
            if (code == EOF || code == EXPIRED) {
                return new KeyPressed("ESC");
            }
            if (code == 'A') {
                return new KeyPressed("UP");
            }
            if (code == 'B') {
                return new KeyPressed("DOWN");
            }
            if (code == 'C') {
                return new KeyPressed("RIGHT");
            }
            if (code == 'D') {
                return new KeyPressed("LEFT");
            }
            if (code == 'Z') {
                return new KeyPressed("SHIFT_TAB");
            }

            if (Character.isDigit(code) || Character.isLetter(code)) {
                final StringBuilder digits = new StringBuilder();
                digits.append((char) code);
                int ch;
                while ((ch = next(ESC_TIMEOUT_MS)) != '~' && ch != EOF && ch != EXPIRED) {
                    digits.append((char) ch);
                }
                return switch (digits.toString()) {
                    case "1", "7", "H" -> new KeyPressed("HOME");
                    case "2" -> new KeyPressed("INSERT");
                    case "3" -> new KeyPressed("DELETE");
                    case "4", "8" -> new KeyPressed("END");
                    case "5" -> new KeyPressed("PAGE_UP");
                    case "6" -> new KeyPressed("PAGE_DOWN");
                    case "200" -> new Pasted(readBracketedPaste());
                    case "P", "11" -> new KeyPressed("FN_1");
                    case "Q" -> new KeyPressed("FN_2");
                    case "R" -> new KeyPressed("FN_3");
                    default -> new KeyPressed("UNKNOWN");
                };
            }
        }

        return new KeyPressed("ALT_" + (char) next);
    }

    /**
     * Read a bracketed paste sequence (ESC[200~...ESC[201~).
     *
     * @return The paste content with normalized line endings, or empty string if none
     * @throws IOException If an I/O error occurs
     */
    private String readBracketedPaste() throws IOException {
        final StringBuilder buf = new StringBuilder();
        while (true) {
            final int c = next(-1);
            if (c == EOF || c == EXPIRED) {
                return buf.toString();
            }
            if (c != ESC) {
                buf.append((char) c);
                continue;
            }
            final int nxt = next(ESC_TIMEOUT_MS);
            if (nxt != CSI_BRACKET) {
                buf.append((char) ESC);
                if (nxt != EOF && nxt != EXPIRED) {
                    buf.append((char) nxt);
                }
                continue;
            }
            final StringBuilder digits = new StringBuilder();
            int ch;
            while ((ch = next(ESC_TIMEOUT_MS)) != '~' && ch != EOF && ch != EXPIRED) {
                digits.append((char) ch);
            }
            if ("201".contentEquals(digits)) {
                return buf.toString();
            }
            buf.append((char) ESC).append((char) CSI_BRACKET).append(digits);
            if (ch == '~') {
                buf.append('~');
            }
        }
    }

    /**
     * Read the next byte, returning cached bytes if any.
     *
     * @param timeoutMs The timeout for reading (ignored if cached byte exists)
     * @return The byte read, or -1 for EOF, -2 for timeout
     * @throws IOException If an I/O error occurs
     */
    private int next(final int timeoutMs) throws IOException {
        if (unread != NONE) {
            final int value = unread;
            unread = NONE;
            return value;
        }
        return source.read(timeoutMs);
    }

    /**
     * Cache a byte to be read on the next call to {@link #next(int)}.
     *
     * @param c The byte to cache
     */
    private void unread(final int c) {
        unread = c;
    }

    /**
     * Append a printable character to the buffer, normalizing line endings.
     *
     * @param buf The StringBuilder to append to
     * @param c The character code to append
     */
    private static void appendPasteChar(final StringBuilder buf, final int c) {
        if (c == 13) {
            buf.append('\r');
        } else if (c == 10) {
            buf.append('\n');
        } else {
            buf.append((char) c);
        }
    }

    /**
     * Check if the byte is a printable character that could be part of a paste.
     *
     * @param c The byte to check
     * @return true if the byte is a valid paste character
     */
    private static boolean isPasteChar(final int c) {
        return c == 9 || c == 10 || c == 13 || (c >= 32 && c != 127);
    }

    /**
     * Convert a byte to its control key name (for Ctrl chars, special keys).
     *
     * Reference:
     * case 0  -> "CTRL_2"; // NUL / Ctrl+@
     * case 1  -> "CTRL_A";
     * case 2  -> "CTRL_B";
     * case 3  -> "CTRL_C";
     * case 4  -> "CTRL_D";
     * case 5  -> "CTRL_E";
     * case 6  -> "CTRL_F";
     * case 7  -> "CTRL_G";
     * case 8  -> "CTRL_H"; // often BACKSPACE
     * case 9  -> "CTRL_I"; // often TAB
     * case 10 -> "CTRL_J";
     * case 11 -> "CTRL_K";
     * case 12 -> "CTRL_L";
     * case 13 -> "CTRL_M"; // often ENTER
     * case 14 -> "CTRL_N";
     * case 15 -> "CTRL_O";
     * case 16 -> "CTRL_P";
     * case 17 -> "CTRL_Q";
     * case 18 -> "CTRL_R";
     * case 19 -> "CTRL_S";
     * case 20 -> "CTRL_T";
     * case 21 -> "CTRL_U";
     * case 22 -> "CTRL_V";
     * case 23 -> "CTRL_W";
     * case 24 -> "CTRL_X";
     * case 25 -> "CTRL_Y";
     * case 26 -> "CTRL_Z";
     * case 27 -> "CTRL_3"; // ESC
     * case 28 -> "CTRL_4";
     * case 29 -> "CTRL_5";
     * case 30 -> "CTRL_6";
     * case 31 -> "CTRL_7";
     * case 127 -> "CTRL_8"; // DEL
     *
     * @param c The byte to convert
     * @return The control key name string
     */
    private static String controlKey(final int c) {
        return switch (c) {
            case 13 -> "ENTER";
            case 10 -> "CTRL_J";
            case 127, 8 -> "BACKSPACE";
            case 9 -> "TAB";
            case 3 -> "CTRL_C";
            case 5 -> "CTRL_E";
            case 19 -> "CTRL_S";
            default -> String.valueOf((char) c);
        };
    }
}
