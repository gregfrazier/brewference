package com.epicmonstrosity.tui.text;

import com.epicmonstrosity.tui.ansi.DisplayWidth;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A model for viewing text files with search and follow-tail support.
 */
public final class TextViewerModel {
    private final List<String> sourceLines = new ArrayList<>();
    private int scrollOffset;
    private String query = "";
    private int matchIndex = -1;
    private boolean followTail;
    private boolean followable;
    private boolean openLine;
    private Path tailPath;
    private long tailOffset;

    /**
     * Construct a text viewer model.
     *
     * @param sourceLines The source lines
     */
    private TextViewerModel(final List<String> sourceLines) {
        if (sourceLines != null) {
            for (String line : sourceLines) {
                this.sourceLines.add(line == null ? "" : line);
            }
        }
    }

    /**
     * Create a model from text.
     *
     * @param text The text
     * @return The TextViewerModel
     */
    public static TextViewerModel from(final String text) {
        if (text == null || text.isEmpty()) {
            return fromLines(List.of());
        }
        return fromLines(List.of(text.replace("\r\n", "\n").replace('\r', '\n').split("\n", -1)));
    }

    /**
     * Create a model from lines.
     *
     * @param lines The lines
     * @return The TextViewerModel
     */
    public static TextViewerModel fromLines(final List<String> lines) {
        return new TextViewerModel(lines == null ? List.of() : lines);
    }

    /**
     * Create a model from a file path.
     *
     * @param path The file path
     * @return The TextViewerModel
     * @throws IOException If reading fails
     */
    public static TextViewerModel fromPath(final Path path) throws IOException {
        return from(Files.readString(path));
    }

    /**
     * Create a tailing model that reads from a file as it grows.
     *
     * @param path The file path
     * @return The TextViewerModel
     */
    public static TextViewerModel tail(final Path path) {
        final TextViewerModel model = new TextViewerModel(List.of());
        model.tailPath = path;
        model.followable = true;
        model.followTail = true;
        model.poll();
        return model;
    }

    /**
     * Enable following the tail.
     *
     * @return This model
     */
    public synchronized TextViewerModel followTail() {
        return followTail(true);
    }

    /**
     * Enable or disable following the tail.
     *
     * @param follow Whether to follow
     * @return This model
     */
    public synchronized TextViewerModel followTail(final boolean follow) {
        this.followable = true;
        this.followTail = follow;
        return this;
    }

    /**
     * Check if following the tail.
     *
     * @return true if following
     */
    public synchronized boolean following() {
        return followTail;
    }

    /**
     * Check if following is enabled.
     *
     * @return true if followable
     */
    public synchronized boolean followable() {
        return followable;
    }

    /**
     * Append text to the model.
     *
     * @param text The text to append
     */
    public synchronized void append(final String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        followable = true;
        String normalized = text.replace("\r\n", "\n").replace('\r', '\n');
        if (openLine && !sourceLines.isEmpty()) {
            normalized = sourceLines.removeLast() + normalized;
        }
        final boolean endsWithNewline = normalized.endsWith("\n");
        final String[] parts = normalized.split("\n", -1);
        final int last = parts.length - 1;
        for (int i = 0; i < last; i++) {
            sourceLines.add(parts[i]);
        }
        if (!endsWithNewline) {
            sourceLines.add(parts[last]);
        }
        openLine = !endsWithNewline;
    }

    /**
     * Append multiple lines to the model.
     *
     * @param lines The lines to append
     */
    public synchronized void appendLines(final List<String> lines) {
        if (lines == null || lines.isEmpty()) {
            return;
        }
        for (String line : lines) {
            append((line == null ? "" : line) + "\n");
        }
    }

    /**
     * Read newly appended bytes from a tail file.
     *
     * @return true if any text was ingested
     */
    public synchronized boolean poll() {
        if (tailPath == null) {
            return false;
        }
        try {
            if (!Files.exists(tailPath)) {
                return false;
            }
            final long size = Files.size(tailPath);
            if (size < tailOffset) {
                tailOffset = 0;
                sourceLines.clear();
                openLine = false;
            }
            if (size <= tailOffset) {
                return false;
            }
            final int n = (int) Math.min(size - tailOffset, 1_048_576);
            final ByteBuffer buffer = ByteBuffer.allocate(n);
            try (var channel = Files.newByteChannel(tailPath, StandardOpenOption.READ)) {
                channel.position(tailOffset);
                while (buffer.hasRemaining()) {
                    final int read = channel.read(buffer);
                    if (read < 0) {
                        break;
                    }
                }
                tailOffset = channel.position();
            }
            buffer.flip();
            if (!buffer.hasRemaining()) {
                return false;
            }
            final byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            append(new String(bytes, StandardCharsets.UTF_8));
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    /**
     * Get the scroll offset.
     *
     * @return The scroll offset
     */
    public synchronized int scrollOffset() {
        return scrollOffset;
    }

    /**
     * Get the source lines.
     *
     * @return The source lines
     */
    public synchronized List<String> sourceLines() {
        return List.copyOf(sourceLines);
    }

    /**
     * Get the search query.
     *
     * @return The query
     */
    public synchronized String query() {
        return query;
    }

    /**
     * Get the match index.
     *
     * @return The match index
     */
    public synchronized int matchIndex() {
        return matchIndex;
    }

    /**
     * Set the search query.
     *
     * @param query The query
     */
    public synchronized void setQuery(final String query) {
        this.query = query == null ? "" : query;
        this.matchIndex = -1;
    }

    /**
     * Type a character into the search query.
     *
     * @param text The character
     */
    public synchronized void typeQuery(final String text) {
        if (text != null && !text.isEmpty()) {
            query += text;
            matchIndex = -1;
        }
    }

    /**
     * Delete the last character from the search query.
     */
    public synchronized void backspaceQuery() {
        if (!query.isEmpty()) {
            query = query.substring(0, query.length() - 1);
            matchIndex = -1;
        }
    }

    /**
     * Clear the search query.
     */
    public synchronized void clearSearch() {
        query = "";
        matchIndex = -1;
    }

    /**
     * Find the next match.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     * @return true if a match was found
     */
    public synchronized boolean findNext(final int terminalCols, final int viewportHeight) {
        return find(terminalCols, viewportHeight, 1);
    }

    /**
     * Find the previous match.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     * @return true if a match was found
     */
    public synchronized boolean findPrevious(final int terminalCols, final int viewportHeight) {
        return find(terminalCols, viewportHeight, -1);
    }

    /**
     * Wrap all lines to the given columns.
     *
     * @param terminalCols The terminal columns
     * @return The wrapped lines
     */
    public synchronized List<String> wrap(final int terminalCols) {
        final int cols = Math.max(1, terminalCols);
        final List<String> wrapped = new ArrayList<>();
        for (String line : sourceLines) {
            if (line == null || line.isEmpty()) {
                wrapped.add("");
                continue;
            }
            wrapped.addAll(DisplayWidth.wrap(line, cols));
        }
        return wrapped;
    }

    /**
     * Get the visible window of wrapped lines.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     * @return The visible lines
     */
    public synchronized List<String> visibleWindow(final int terminalCols, final int viewportHeight) {
        final List<String> wrapped = wrap(terminalCols);
        if (viewportHeight <= 0 || wrapped.isEmpty()) {
            return List.of();
        }
        final int max = maxOffset(wrapped.size(), viewportHeight);
        if (followTail) {
            scrollOffset = max;
        } else {
            scrollOffset = clamp(scrollOffset, 0, max);
        }
        final int to = Math.min(wrapped.size(), scrollOffset + viewportHeight);
        return List.copyOf(wrapped.subList(scrollOffset, to));
    }

    /**
     * Move the scroll offset by a delta.
     *
     * @param delta The delta to move
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     */
    public synchronized void moveBy(final int delta, final int terminalCols, final int viewportHeight) {
        final int total = wrap(terminalCols).size();
        final int max = maxOffset(total, viewportHeight);
        if (followTail) {
            scrollOffset = max;
        }
        scrollOffset = clamp(scrollOffset + delta, 0, max);
        followTail = followable && scrollOffset >= max;
    }

    /**
     * Page up the scroll offset.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     */
    public synchronized void pageUp(final int terminalCols, final int viewportHeight) {
        moveBy(-Math.max(1, viewportHeight), terminalCols, viewportHeight);
    }

    /**
     * Page down the scroll offset.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     */
    public synchronized void pageDown(final int terminalCols, final int viewportHeight) {
        moveBy(Math.max(1, viewportHeight), terminalCols, viewportHeight);
    }

    /**
     * Move to the top.
     */
    public synchronized void home() {
        scrollOffset = 0;
        followTail = false;
    }

    /**
     * Move to the bottom.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     */
    public synchronized void end(final int terminalCols, final int viewportHeight) {
        scrollOffset = maxOffset(wrap(terminalCols).size(), viewportHeight);
        if (followable) {
            followTail = true;
        }
    }

    /**
     * Get the total number of wrapped lines.
     *
     * @param terminalCols The terminal columns
     * @return The total wrapped lines
     */
    public synchronized int totalWrapped(final int terminalCols) {
        return wrap(terminalCols).size();
    }

    /**
     * Calculate the maximum scroll offset.
     *
     * @param total The total lines
     * @param viewportHeight The viewport height
     * @return The maximum offset
     */
    static int maxOffset(final int total, final int viewportHeight) {
        return Math.max(0, total - Math.max(1, viewportHeight));
    }

    /**
     * Find a match in a direction.
     *
     * @param terminalCols The terminal columns
     * @param viewportHeight The viewport height
     * @param direction The search direction (1 for next, -1 for previous)
     * @return true if a match was found
     */
    private boolean find(final int terminalCols, final int viewportHeight, final int direction) {
        if (query.isBlank()) {
            return false;
        }
        final List<String> wrapped = wrap(terminalCols);
        if (wrapped.isEmpty()) {
            return false;
        }
        final String needle = query.toLowerCase(Locale.ROOT);
        final int start = matchIndex < 0
                ? (direction > 0 ? 0 : wrapped.size() - 1)
                : Math.floorMod(matchIndex + direction, wrapped.size());
        for (int i = 0; i < wrapped.size(); i++) {
            final int idx = Math.floorMod(start + i * direction, wrapped.size());
            final String line = wrapped.get(idx);
            if (line != null && line.toLowerCase(Locale.ROOT).contains(needle)) {
                matchIndex = idx;
                ensureVisible(viewportHeight, wrapped.size());
                followTail = followable && scrollOffset >= maxOffset(wrapped.size(), viewportHeight);
                return true;
            }
        }
        return false;
    }

    /**
     * Ensure the match is visible.
     *
     * @param viewportHeight The viewport height
     * @param total The total lines
     */
    private void ensureVisible(final int viewportHeight, final int total) {
        final int height = Math.max(1, viewportHeight);
        if (matchIndex < scrollOffset || matchIndex >= scrollOffset + height) {
            scrollOffset = clamp(matchIndex, 0, maxOffset(total, height));
        }
    }

    /**
     * Clamp a value to a range.
     *
     * @param value The value to clamp
     * @param min The minimum
     * @param max The maximum
     * @return The clamped value
     */
    private static int clamp(final int value, final int min, final int max) {
        return Math.max(min, Math.min(max, value));
    }
}
