package com.epicmonstrosity.tui;

/** Emitted by {@link TuiApp} when the terminal size changes. */
public record TerminalResized(int cols, int rows) implements UiMessage {
}
