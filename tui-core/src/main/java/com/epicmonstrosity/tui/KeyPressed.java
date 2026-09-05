package com.epicmonstrosity.tui;

/** A key was pressed in the terminal. */
public record KeyPressed(String key) implements UiMessage {
}
