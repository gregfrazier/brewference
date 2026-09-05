package com.epicmonstrosity.tui.screen;

/**
 * A screen layout containing header, body, status line, and footer error.
 */
public record ScreenLayout(String header,
                           String body,
                           String statusLine,
                           String footerError) {
}
