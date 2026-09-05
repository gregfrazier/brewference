package com.epicmonstrosity.tui.ansi;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DisplayWidthTest {
    @Test
    void asciiIsOneColumnEach() {
        assertEquals(1, DisplayWidth.of('A'));
        assertEquals(5, DisplayWidth.of("hello"));
    }

    @Test
    void cjkAndFullwidthAreTwoColumns() {
        assertEquals(2, DisplayWidth.of("你"));
        assertEquals(4, DisplayWidth.of("你好"));
        assertEquals(2, DisplayWidth.of("Ａ"));
        assertEquals(2, DisplayWidth.of("한"));
    }

    @Test
    void emojiIsTwoColumnsAndCombiningMarksAreZero() {
        assertEquals(2, DisplayWidth.of("👋"));
        assertEquals(1, DisplayWidth.of("e\u0301"));
        assertEquals(4, DisplayWidth.of("a👋b"));
    }

    @Test
    void controlsAndFormatAreZeroWidth() {
        assertEquals(0, DisplayWidth.of('\n'));
        assertEquals(0, DisplayWidth.of('\u0007'));
        assertEquals(0, DisplayWidth.of('\u200B'));
        assertEquals(0, DisplayWidth.of('\uFE0F'));
    }

    @Test
    void prefixStopsBeforeAWideGlyphThatWouldOverflow() {
        assertEquals("你", DisplayWidth.prefix("你好", 3));
        assertEquals("你好", DisplayWidth.prefix("你好", 4));
        assertEquals("a", DisplayWidth.prefix("a👋b", 2));
        assertEquals("a👋", DisplayWidth.prefix("a👋b", 3));
    }

    @Test
    void wrapBreaksOnDisplayColumnsNotJavaChars() {
        assertEquals(List.of("你好", "世界"), DisplayWidth.wrap("你好世界", 4));
        assertEquals(List.of("a你", "b"), DisplayWidth.wrap("a你b", 3));
        assertEquals(List.of(""), DisplayWidth.wrap("", 8));
    }
}
