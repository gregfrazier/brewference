package com.epicmonstrosity.brewference.cli.screen.chat;

import com.epicmonstrosity.tui.ansi.DisplayWidth;

import java.util.ArrayList;
import java.util.List;

/**
 * The chat model managing messages, input, scrolling, and rendering.
 */
public final class ChatModel {
    public static final int GUTTER_WIDTH = 9;
    public static final long CHURN_FRAME_NANOS = 80_000_000L;
    static final String[] CHURN_FRAMES = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

    private final List<ChatMessage> messages = new ArrayList<>();
    private final StringBuilder inputBuffer = new StringBuilder();
    private int scrollOffset;
    private boolean stickToBottom = true;
    private int lastCols = 80;

    /**
     * Get the list of chat messages (unmodifiable copy).
     *
     * @return Unmodifiable list of messages
     */
    public synchronized List<ChatMessage> messages() {
        return List.copyOf(messages);
    }

    /**
     * Get the current scroll offset.
     *
     * @return The scroll offset
     */
    public synchronized int scrollOffset() {
        return scrollOffset;
    }

    /**
     * Check if the view should stick to the bottom.
     *
     * @return true if stuck to bottom
     */
    public synchronized boolean stickToBottom() {
        return stickToBottom;
    }

    /**
     * Get the current input buffer.
     *
     * @return The input buffer content
     */
    public synchronized String inputBuffer() {
        return inputBuffer.toString();
    }

    /**
     * Check if any message is currently streaming.
     *
     * @return true if a message is streaming
     */
    public synchronized boolean streaming() {
        return lastMessageIsStreaming();
    }

    /**
     * Check if the model is in a generating state (assistant streaming empty text).
     *
     * @return true if generating
     */
    public synchronized boolean generating() {
        final ChatMessage last = lastMessage();
        return last != null && last.streaming() && isEmpty(last.text());
    }

    /**
     * Append a message to the chat.
     *
     * @param message The message to append
     */
    public synchronized void append(final ChatMessage message) {
        messages.add(message);
    }

    /**
     * Start generating (streaming) with default assistant role.
     */
    public synchronized void startGenerating() {
        startStreaming(ChatMessage.ASSISTANT);
    }

    /**
     * Start generating with a custom role.
     *
     * @param role The role for the generating message
     */
    public synchronized void startGenerating(final String role) {
        startStreaming(role);
    }

    /**
     * Start streaming a message with the given role.
     *
     * @param role The role for the streaming message
     */
    public synchronized void startStreaming(final String role) {
        messages.add(new ChatMessage(role == null ? ChatMessage.ASSISTANT : role, "", java.time.Instant.now(), true));
    }

    /**
     * Append a delta of text to the current streaming message.
     *
     * @param delta The text delta to append
     */
    public synchronized void appendDelta(final String delta) {
        if (delta == null || delta.isEmpty()) {
            return;
        }
        if (!lastMessageIsStreaming()) {
            startStreaming(ChatMessage.ASSISTANT);
        }
        final ChatMessage last = messages.getLast();
        messages.set(messages.size() - 1, last.withText(textOrEmpty(last.text()) + delta));
    }

    /**
     * Finish the current streaming message.
     */
    public synchronized void finishStreaming() {
        if (messages.isEmpty()) {
            return;
        }
        final ChatMessage last = messages.getLast();
        if (last.streaming()) {
            messages.set(messages.size() - 1, last.finished());
        }
    }

    /**
     * Type a character into the input buffer.
     *
     * @param text The character to type
     */
    public synchronized void type(final String text) {
        if (text != null) {
            inputBuffer.append(text);
        }
    }

    /**
     * Paste text into the input buffer.
     *
     * @param text The text to paste
     */
    public synchronized void paste(final String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        inputBuffer.append(text.replace("\r\n", "\n").replace('\r', '\n'));
    }

    /**
     * Add a newline to the input buffer.
     */
    public synchronized void newline() {
        inputBuffer.append('\n');
    }

    /**
     * Delete the last character from the input buffer.
     */
    public synchronized void backspace() {
        if (!inputBuffer.isEmpty()) {
            inputBuffer.deleteCharAt(inputBuffer.length() - 1);
        }
    }

    /**
     * Take the input buffer and clear it.
     *
     * @return The input buffer content
     */
    public synchronized String takeInput() {
        final String text = inputBuffer.toString();
        inputBuffer.setLength(0);
        return text;
    }

    /**
     * Submit user text as a message to the chat.
     *
     * @param text The text to submit
     */
    public synchronized void submitUser(final String text) {
        if (text == null || text.isBlank()) {
            return;
        }
        append(ChatMessage.user(text));
    }

    /**
     * Clear all messages and reset input and scroll state.
     */
    public synchronized void clear() {
        messages.clear();
        inputBuffer.setLength(0);
        scrollOffset = 0;
        stickToBottom = true;
    }

    /**
     * Scroll by a delta (viewportHeight version).
     *
     * @param delta The scroll delta
     * @param viewportHeight The viewport height
     */
    public synchronized void scrollBy(final int delta, final int viewportHeight) {
        scrollBy(delta, lastCols, viewportHeight);
    }

    /**
     * Scroll by a delta.
     *
     * @param delta The scroll delta
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     */
    public synchronized void scrollBy(final int delta, final int cols, final int viewportHeight) {
        lastCols = positive(cols);
        final int max = maxLineOffset(lastCols, viewportHeight);
        if (stickToBottom) {
            scrollOffset = max;
        }
        scrollOffset = clamp(safeAdd(scrollOffset, delta), 0, max);
        stickToBottom = scrollOffset >= max;
    }

    /**
     * Page up the chat.
     *
     * @param viewportHeight The viewport height
     */
    public synchronized void pageUp(final int viewportHeight) {
        scrollBy(-Math.max(1, viewportHeight), lastCols, viewportHeight);
    }

    /**
     * Page down the chat.
     *
     * @param viewportHeight The viewport height
     */
    public synchronized void pageDown(final int viewportHeight) {
        scrollBy(Math.max(1, viewportHeight), lastCols, viewportHeight);
    }

    /**
     * Get the visible window of messages.
     *
     * @param viewportHeight The viewport height
     * @return List of visible messages
     */
    public synchronized List<ChatMessage> visibleWindow(final int viewportHeight) {
        if (viewportHeight <= 0 || messages.isEmpty()) {
            return List.of();
        }
        final int max = maxOffset(viewportHeight);
        final int from = stickToBottom ? max : Math.min(scrollOffset, max);
        scrollOffset = from;
        final int to = Math.min(messages.size(), safeAdd(from, viewportHeight));
        return List.copyOf(messages.subList(from, to));
    }

    /**
     * Get the visible lines as formatted strings (system time).
     *
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     * @return List of formatted visible lines
     */
    public synchronized List<String> visibleLines(final int cols, final int viewportHeight) {
        return visibleLines(cols, viewportHeight, System.nanoTime());
    }

    /**
     * Get the visible lines as formatted strings.
     *
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     * @param nowNanos The current time in nanoseconds
     * @return List of formatted visible lines
     */
    public synchronized List<String> visibleLines(final int cols, final int viewportHeight, final long nowNanos) {
        return visibleChatLines(cols, viewportHeight, nowNanos).stream().map(ChatLine::formatted).toList();
    }

    /**
     * Get the visible chat lines (system time).
     *
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     * @return List of visible ChatLine objects
     */
    public synchronized List<ChatLine> visibleChatLines(final int cols, final int viewportHeight) {
        return visibleChatLines(cols, viewportHeight, System.nanoTime());
    }

    /**
     * Get the visible chat lines.
     *
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     * @param nowNanos The current time in nanoseconds
     * @return List of visible ChatLine objects
     */
    public synchronized List<ChatLine> visibleChatLines(final int cols, final int viewportHeight, final long nowNanos) {
        lastCols = positive(cols);
        final List<ChatLine> lines = wrappedChatLines(lastCols, nowNanos);
        if (viewportHeight <= 0 || lines.isEmpty()) {
            return List.of();
        }
        final int max = maxOffset(lines.size(), viewportHeight);
        scrollOffset = stickToBottom ? max : clamp(scrollOffset, 0, max);
        final int to = Math.min(lines.size(), safeAdd(scrollOffset, viewportHeight));
        return List.copyOf(lines.subList(scrollOffset, to));
    }

    /**
     * Get the total number of wrapped lines.
     *
     * @param cols The terminal columns
     * @return The total wrapped line count
     */
    public synchronized int totalWrapped(final int cols) {
        return wrappedChatLines(positive(cols), System.nanoTime()).size();
    }

    /**
     * Get the churn text (generating animation).
     *
     * @param nowNanos The current time in nanoseconds
     * @return The churn text
     */
    public String churnText(final long nowNanos) {
        final int frame = (int) Math.floorMod(nowNanos / CHURN_FRAME_NANOS, CHURN_FRAMES.length);
        return CHURN_FRAMES[frame] + " generating";
    }

    /**
     * Calculate the maximum scroll offset.
     *
     * @param viewportHeight The viewport height
     * @return The maximum offset
     */
    int maxOffset(final int viewportHeight) {
        return maxOffset(messages.size(), viewportHeight);
    }

    /**
     * Calculate the line maximum offset.
     *
     * @param cols The terminal columns
     * @param viewportHeight The viewport height
     * @return The maximum offset
     */
    private int maxLineOffset(final int cols, final int viewportHeight) {
        return maxOffset(wrappedChatLines(cols, System.nanoTime()).size(), viewportHeight);
    }

    /**
     * Wrap all chat lines.
     *
     * @param cols The terminal columns
     * @param nowNanos The current time in nanoseconds
     * @return List of wrapped ChatLine objects
     */
    private List<ChatLine> wrappedChatLines(final int cols, final long nowNanos) {
        final List<ChatLine> lines = new ArrayList<>();
        for (final ChatMessage message : messages) {
            paintMessage(lines, message, cols, nowNanos);
        }
        return lines;
    }

    /**
     * Paint a message into the output list.
     *
     * @param out The output list to append to
     * @param message The message to paint
     * @param cols The terminal columns
     * @param nowNanos The current time in nanoseconds
     */
    private void paintMessage(final List<ChatLine> out, final ChatMessage message, final int cols, final long nowNanos) {
        final String name = roleName(message.role());
        final String gutter = pad(name, GUTTER_WIDTH);
        final String empty = " ".repeat(GUTTER_WIDTH);
        String body = textOrEmpty(message.text());
        if (message.streaming() && body.isEmpty()) {
            body = churnText(nowNanos);
        } else if (message.streaming()) {
            body += "▌";
        }
        final int bodyCols = Math.max(1, cols - GUTTER_WIDTH - ChatLine.SEPARATOR.length());
        final List<String> wrapped = new ArrayList<>();
        wrapInto(wrapped, body, bodyCols);
        boolean first = true;
        for (final String line : wrapped) {
            out.add(new ChatLine(message.role(), first ? gutter : empty, line, !first));
            first = false;
        }
    }

    /**
     * Get the role name (lowercase).
     *
     * @param role The role
     * @return The role name
     */
    static String roleName(final String role) {
        return switch (role == null ? "" : role) {
            case ChatMessage.ASSISTANT -> "assistant";
            case ChatMessage.SYSTEM -> "system";
            case ChatMessage.TOOL -> "tool";
            default -> "user";
        };
    }

    /**
     * Get the role label with colon.
     *
     * @param role The role
     * @return The role label
     */
    static String roleLabel(final String role) {
        return roleName(role) + ":";
    }

    /**
     * Wrap text into paragraphs and lines.
     *
     * @param out The output list to append to
     * @param text The text to wrap
     * @param cols The terminal columns
     */
    static void wrapInto(final List<String> out, final String text, final int cols) {
        final String[] paragraphs = text.replace("\r\n", "\n").replace('\r', '\n').split("\n", -1);
        for (final String paragraph : paragraphs) {
            if (paragraph.isEmpty()) {
                out.add("");
                continue;
            }
            out.addAll(DisplayWidth.wrap(paragraph, cols));
        }
    }

    /**
     * Pad text to the given width.
     *
     * @param text The text to pad
     * @param width The target width
     * @return The padded text
     */
    private static String pad(final String text, final int width) {
        final String value = text == null ? "" : text;
        final int columns = DisplayWidth.of(value);
        if (columns >= width) {
            return DisplayWidth.prefix(value, width);
        }
        return value + " ".repeat(width - columns);
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

    private ChatMessage lastMessage() {
        return messages.isEmpty() ? null : messages.getLast();
    }

    private boolean lastMessageIsStreaming() {
        final ChatMessage last = lastMessage();
        return last != null && last.streaming();
    }

    private static int maxOffset(final int itemCount, final int viewportHeight) {
        return Math.max(0, itemCount - positive(viewportHeight));
    }

    private static int positive(final int value) {
        return Math.max(1, value);
    }

    private static String textOrEmpty(final String text) {
        return text == null ? "" : text;
    }

    private static boolean isEmpty(final String text) {
        return text == null || text.isEmpty();
    }

    private static int safeAdd(final int left, final int right) {
        return clamp(left + right, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
}
