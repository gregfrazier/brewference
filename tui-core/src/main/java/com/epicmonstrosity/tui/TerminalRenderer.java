package com.epicmonstrosity.tui;

import com.epicmonstrosity.tui.ansi.Ansi;
import com.epicmonstrosity.tui.overlay.Toast;
import com.epicmonstrosity.tui.overlay.ToastBanner;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Owns the JLine terminal handle and is responsible for:
 *  - entering / leaving raw mode
 *  - querying terminal dimensions on every render
 *  - writing frames so the footer always appears on the last visible row
 * <p>
 * How the layout works
 * ─────────────────────
 * Every render splits the output into three parts:
 * <p>
 *   ┌─────────────────────────┐  ← row 1
 *   │  header                 │
 *   │  scrollable body rows   │  ← [headerRows .. termRows - footerRows - 1]
 *   │                         │
 *   ├─────────────────────────┤  ← row (termRows - footerRows)
 *   │  sticky footer          │  ← always last N rows
 *   └─────────────────────────┘  ← row termRows
 */
public class TerminalRenderer implements AutoCloseable, TerminalRect {

    private final Terminal terminal;
    private final PrintWriter out;
    private final InputDecoder decoder;

    public static final String HORIZONTAL_LINE_CHAR = "─";
    public static final String HIDE_CURSOR = "\u001B[?25l";
    public static final String SHOW_CURSOR = "\u001B[?25h";
    public static final String CLEAR_SCREEN = "\u001B[2J";
    public static final String HOME = "\u001B[H";
    public static final String ERASE_LINE = "\u001B[2K";
    public static final String MOVE = "\u001B[%d;1H";

    private static final int READ_TIMEOUT_MS = 100;
    private static final String ENABLE_BRACKETED_PASTE = "\u001B[?2004h";
    private static final String DISABLE_BRACKETED_PASTE = "\u001B[?2004l";

    /** Number of rows the footer occupies (controls bar + optional error banner) */
    public static final int FOOTER_ROWS = 2;

    /** Number of rows the header occupies (title + underline + blank) */
    public static final int HEADER_ROWS = 5;

    /**
     * Constructs a new TerminalRenderer. Initializes the JLine terminal in raw mode,
     * sets up the output stream, and prepares the decoder.
     * 
     * @throws IOException If the terminal cannot be initialized.
     */
    public TerminalRenderer() throws IOException {
        this.terminal = TerminalBuilder.builder()
                .system(true)
                .jansi(false)
                .build();
        this.terminal.enterRawMode();
        this.out = terminal.writer();
        this.decoder = new InputDecoder(timeoutMs -> terminal.reader().read(timeoutMs));
        out.print(HIDE_CURSOR);
        out.print(ENABLE_BRACKETED_PASTE);
        out.flush();
    }

    /**
     * Read the next terminal event: a key, a bracketed paste, or {@code null} on EOF.
     * Calls {@code invalidatedRender} when input expires with no data (used to trigger redraws).
     *
     * @param invalidatedRender Callback to run when idle timeout expires
     * @return The next UI event, or null for EOF
     * @throws IOException If an I/O error occurs
     */
    public UiMessage readEvent(final Runnable invalidatedRender) throws IOException {
        return decoder.readEvent(READ_TIMEOUT_MS, invalidatedRender);
    }

    /**
     * Renders the given screen with no overlay or toast.
     * 
     * @param screen The screen content to render.
     */
    public void render(final Screen screen) {
        render(screen, null, null);
    }

    /**
     * Renders the given screen with an overlay and no toast.
     * 
     * @param screen   The screen content to render.
     * @param overlay  The overlay to apply over the screen.
     */
    public void render(final Screen screen, final Overlay overlay) {
        render(screen, overlay, null);
    }

    /**
     * Renders the given screen with an overlay and a single toast notification.
     * 
     * @param screen  The screen content to render.
     * @param overlay The overlay to apply over the screen.
     * @param toast   The toast notification to display, or null for none
     */
    public void render(final Screen screen, final Overlay overlay, final Toast toast) {
        render(screen, overlay, toast == null ? List.of() : List.of(toast), null, 0);
    }

    /**
     * Renders the given screen with an overlay and a single toast notification,
     * tracking the number of pending toasts.
     * 
     * @param screen         The screen content to render.
     * @param overlay        The overlay to apply over the screen.
     * @param toast          The toast notification to display, or null for none
     * @param pendingToasts  The number of toasts currently pending
     */
    public void render(final Screen screen, final Overlay overlay, final Toast toast, final int pendingToasts) {
        render(screen, overlay, toast == null ? List.of() : List.of(toast), null, pendingToasts);
    }

    /**
     * Performs the full rendering of the screen, overlay, toasts, and status.
     * This is the primary render method that composes all UI components.
     * 
     * @param screen         The screen content to render
     * @param overlay        The overlay to apply over the screen, or null for none
     * @param toasts         A list of toast notifications to display
     * @param status         The sticky status toast to display, or null for none
     * @param pendingToasts  The number of toasts currently pending
     */
    public void render(final Screen screen,
                       final Overlay overlay,
                       final List<Toast> toasts,
                       final Toast status,
                       final int pendingToasts) {
        final int rows = terminal.getHeight() <= 0 ? 20 : terminal.getHeight();
        final int cols = terminal.getWidth() <= 0 ? 80 : terminal.getWidth();
        if (rows < 8 || cols < 40) {
            renderTooSmall();
            return;
        }

        final int bodyRows = rows - HEADER_ROWS - FOOTER_ROWS;
        final ViewContext ctx = new ViewContext(rows, cols, bodyRows);

        final String[] parts = screen.render(ctx);
        final String header = parts[0];
        String body = parts[1];
        final String footer = parts[2];
        if (overlay != null) {
            body = overlay.renderBody(ctx, body);
        }
        if ((toasts != null && !toasts.isEmpty()) || status != null) {
            body = ToastBanner.paint(body, toasts, status, cols, bodyRows, pendingToasts);
        }

        final StringBuilder frame = new StringBuilder(rows * (cols + 10));
        frame.append(HOME);
        appendHeader(frame, header, cols);
        appendBody(frame, body, bodyRows);
        appendFooter(frame, footer, rows, cols);
        frame.append(Ansi.RESET);

        out.print(frame);
        out.flush();
    }

    /**
     * Renders a fallback message when the terminal dimensions are too small.
     * This handles terminals smaller than 8 rows x 40 columns.
     */
    private void renderTooSmall() {
        out.print(HOME + CLEAR_SCREEN);
        out.print("  Terminal too small — please resize.");
        out.flush();
    }

    /**
     * Appends the header section to the frame buffer.
     * 
     * @param frame  The StringBuilder representing the current frame
     * @param header The header content, or null for none
     * @param cols   The number of columns in the terminal
     */
    public static void appendHeader(final StringBuilder frame, final String header, final int cols) {
        final String[] lines = header == null ? new String[0] : header.split("\n", -1);
        for (int i = 0; i < HEADER_ROWS; i++) {
            final String line = i < lines.length ? lines[i] : "";
            frame.append(ERASE_LINE).append(line).append("\r\n");
        }
    }

    /**
     * Appends the body section to the frame buffer.
     * 
     * @param frame      The StringBuilder representing the current frame
     * @param body       The body content, or null for none
     * @param bodyRows   The number of rows available for the body
     */
    public static void appendBody(final StringBuilder frame, final String body, final int bodyRows) {
        final String[] bodyLines = body.split("\n", -1);
        int writtenBodyLines = 0;
        for (final String line : bodyLines) {
            if (writtenBodyLines >= bodyRows) break;
            frame.append(ERASE_LINE).append(line).append("\r\n");
            writtenBodyLines++;
        }
        while (writtenBodyLines < bodyRows) {
            frame.append(ERASE_LINE).append("\r\n");
            writtenBodyLines++;
        }
    }

    /**
     * Appends the footer section to the frame buffer.
     * 
     * @param frame  The StringBuilder representing the current frame
     * @param footer The footer content, or null for none
     * @param rows   The total number of rows in the terminal
     * @param cols   The number of columns in the terminal
     */
    public static void appendFooter(final StringBuilder frame, final String footer, final int rows, final int cols) {
        final int footerSeparatorRow = rows - FOOTER_ROWS + 1;

        frame.append(String.format(MOVE, footerSeparatorRow));
        frame.append(ERASE_LINE)
                .append("  ")
                .repeat(HORIZONTAL_LINE_CHAR, Math.max(0, cols - 4))
                .append("\r\n");

        frame.append(String.format(MOVE, rows));
        String footerLine = footer.replace("\n", "");
        if (footerLine.length() > cols - 2) {
            footerLine = footerLine.substring(0, cols - 2);
        }
        frame.append(ERASE_LINE).append(footerLine);
    }

    /**
     * Gets the current width of the terminal.
     *
     * @return The terminal width in columns
     */
    @Override
    public int getWidth()  { return terminal.getWidth(); }

    /**
     * Gets the current height of the terminal.
     *
     * @return The terminal height in rows
     */
    @Override
    public int getHeight() { return terminal.getHeight(); }

    /**
     * Closes the terminal, restores the cursor, and clears the screen.
     * 
     * @throws IOException If an I/O error occurs while closing.
     */
    @Override
    public void close() throws IOException {
        out.print(DISABLE_BRACKETED_PASTE);
        out.print(SHOW_CURSOR);
        out.print(HOME + CLEAR_SCREEN);
        out.flush();
        terminal.close();
    }
}
