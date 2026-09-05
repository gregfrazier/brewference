package com.epicmonstrosity.brewference.template.jinja;

import java.util.ArrayList;

/** Shared quote-, escape-, and depth-aware scanning helpers for Jinja sources. */
final class JinjaSourceScanning {
    private JinjaSourceScanning() {}

    /** Finds a token at nesting depth zero, outside strings and nested parentheses or brackets. */
    public static int findTopLevel(final String source, final String token, final int from) {
        var depth = 0;
        char quote = 0;
        var escaped = false;
        for (var index = from; index <= source.length() - token.length(); index++) {
            final var current = source.charAt(index);
            if (quote != 0) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
            } else if (current == '\'' || current == '"') quote = current;
            else if (current == '(' || current == '[') depth++;
            else if (current == ')' || current == ']') depth--;
            else if (depth == 0 && source.startsWith(token, index)) return index;
        }
        return -1;
    }

    /** Splits source at delimiters outside strings and nested groups. */
    public static String[] splitTopLevel(final String source, final char delimiter) {
        final var parts = new ArrayList<String>();
        var start = 0;
        var depth = 0;
        char quote = 0;
        var escaped = false;
        for (var index = 0; index < source.length(); index++) {
            final var current = source.charAt(index);
            if (quote != 0) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
            } else if (current == '\'' || current == '"') quote = current;
            else if (current == '(' || current == '[') depth++;
            else if (current == ')' || current == ']') depth--;
            else if (current == delimiter && depth == 0) {
                parts.add(source.substring(start, index));
                start = index + 1;
            }
        }
        parts.add(source.substring(start));
        return parts.toArray(String[]::new);
    }

    /** Finds a filter pipe that is outside a quoted string. */
    public static int findPipeOutsideString(final CharSequence source, final int from) {
        char quote = 0;
        var escaped = false;
        for (var index = from; index < source.length(); index++) {
            final var current = source.charAt(index);
            if (quote != 0) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
            } else if (current == '\'' || current == '"') quote = current;
            else if (current == '|') return index;
        }
        return -1;
    }

    /** Finds a loop reference outside quoted strings. */
    public static int findLoopReferenceOutsideString(final String source, final int from) {
        char quote = 0;
        var escaped = false;
        for (var index = from; index <= source.length() - 5; index++) {
            final var current = source.charAt(index);
            if (quote != 0) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
            } else if (current == '\'' || current == '"') quote = current;
            else if (source.startsWith("loop.", index)
                    && (index == 0 || !isIdentifierCharacter(source.charAt(index - 1)))) return index;
        }
        return -1;
    }

    /** Determines whether a character belongs to the supported identifier syntax. */
    public static boolean isIdentifierCharacter(final char value) {
        return Character.isLetterOrDigit(value) || value == '_';
    }
}
