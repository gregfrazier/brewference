package com.epicmonstrosity.brewference.template.jinja;

/** Coordinates Jinja source scanning and delegates code generation to focused writers. */
final class JinjaTemplateCompiler {
    private final JinjaExpressionRewriter expressions = new JinjaExpressionRewriter();
    private final JinjaStatementWriter statements = new JinjaStatementWriter(expressions);

    /** Compiles a Jinja template into executable Groovy source. */
    public String compile(final String jinja) {
        if (jinja == null)
            throw new IllegalArgumentException("jinja must not be null");
        statements.reset();
        final var groovy = new StringBuilder(JinjaRuntimePrelude.prelude());
        var position = 0;
        var trimLeadingWhitespace = false;
        var previousTagWasComment = false;
        while (position < jinja.length()) {
            final var tagStart = nextTagStart(jinja, position);
            if (tagStart < 0) {
                final var text = jinja.substring(position);
                if (!previousTagWasComment || !text.isBlank())
                    appendText(groovy, text, trimLeadingWhitespace, false);
                break;
            }
            final var tag = scanTag(jinja, tagStart);
            final var text = jinja.substring(position, tagStart);
            if ((!previousTagWasComment && !tag.comment()) || !text.isBlank())
                appendText(groovy, text, trimLeadingWhitespace, tag.trimBefore());
            appendTag(groovy, tag);
            trimLeadingWhitespace = tag.trimAfter();
            previousTagWasComment = tag.comment();
            position = tag.end();
        }
        if (statements.hasOpenLoop()) throw new IllegalArgumentException("Unclosed Jinja for-loop");
        return groovy.append("out.toString()\n").toString();
    }

    /** A scanned Jinja tag with its whitespace-control flags and end offset. */
    private record Tag(boolean expression, boolean comment, String content, boolean trimBefore, boolean trimAfter, int end) {
    }

    /** Scans one tag at the supplied start offset into a Tag. */
    private Tag scanTag(final String source, final int start) {
        final var expression = source.startsWith("{{", start);
        final var comment = source.startsWith("{#", start);
        final var terminator = expression ? "}}" : comment ? "#}" : "%}";
        final var tagEnd = findTagEnd(source, start + 2, terminator);
        if (tagEnd < 0) throw new IllegalArgumentException("Unterminated Jinja tag at offset " + start);
        final var content = source.substring(start + 2, tagEnd);
        return new Tag(expression, comment, stripTrimFlags(content), content.startsWith("-"), content.endsWith("-"),
                tagEnd + terminator.length());
    }

    /** Removes whitespace-control dashes from a tag's raw content. */
    private String stripTrimFlags(final String content) {
        var stripped = content.startsWith("-") ? content.substring(1) : content;
        if (content.endsWith("-")) stripped = stripped.substring(0, stripped.length() - 1);
        return stripped;
    }

    /** Emits Groovy for one scanned tag. */
    private void appendTag(final StringBuilder groovy, final Tag tag) {
        if (tag.comment()) return;
        if (tag.expression()) {
            groovy.append("out << (").append(expressions.rewrite(tag.content(), statements.loopVariable())).append(")\n");
        } else {
            statements.append(groovy, tag.content().trim());
        }
    }

    /** Finds the next expression or statement tag in the template. */
    private int nextTagStart(final String source, final int from) {
        final var expressionStart = source.indexOf("{{", from);
        final var statementStart = source.indexOf("{%", from);
        final var commentStart = source.indexOf("{#", from);
        var next = expressionStart;
        if (next < 0 || statementStart >= 0 && statementStart < next) next = statementStart;
        if (next < 0 || commentStart >= 0 && commentStart < next) next = commentStart;
        return next;
    }

    /** Finds a tag terminator while ignoring quoted strings. */
    private int findTagEnd(final String source, final int from, final String terminator) {
        char quote = 0;
        var escaped = false;
        for (var index = from; index <= source.length() - terminator.length(); index++) {
            final var current = source.charAt(index);
            if (quote != 0) {
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
            } else if (current == '\'' || current == '"') quote = current;
            else if (source.startsWith(terminator, index)) return index;
        }
        return -1;
    }

    /** Appends escaped template text to the generated output. */
    private void appendText(final StringBuilder groovy, String text, final boolean trimLeading, final boolean trimTrailing) {
        if (trimLeading) text = text.replaceFirst("^\\s+", "");
        if (trimTrailing) text = text.replaceFirst("\\s+$", "");
        if (!text.isEmpty()) groovy.append("out << \"").append(escapeGroovyString(text)).append("\"\n");
    }

    /** Escapes template text for a Groovy double-quoted string. */
    private String escapeGroovyString(final String text) {
        return text.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\r", "\\r").replace("\n", "\\n");
    }
}
