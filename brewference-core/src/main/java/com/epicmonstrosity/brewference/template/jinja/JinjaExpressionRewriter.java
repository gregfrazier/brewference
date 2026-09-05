package com.epicmonstrosity.brewference.template.jinja;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Rewrites the expression subset used by the supported Jinja templates. */
final class JinjaExpressionRewriter {
    private static final Pattern SLICE = Pattern.compile(
            "([A-Za-z_][A-Za-z0-9_.]*)\\[([^:\\]]*):([^:\\]]*)(?::([^]]*))?]"
    );

    /**
     * Applies the supported Jinja-to-Groovy expression transformations.
     *
     * @param rawExpression the source expression
     * @param loopVariable the generated loop metadata variable, or {@code null}
     * @return the rewritten Groovy expression
     */
    public String rewrite(final String rawExpression, final String loopVariable) {
        var expression = rawExpression.trim();
        expression = escapeNewlinesInStringLiterals(expression);
        expression = rewriteStringContains(expression);
        expression = rewriteInlineConditional(expression);
        expression = rewriteFilters(expression);
        expression = rewriteSlices(expression);
        expression = rewriteStringMethods(expression);
        expression = rewriteOperators(expression);
        expression = rewriteDateFunctions(expression);

        return loopVariable == null ? expression : rewriteLoopReference(expression, loopVariable);
    }

    private String rewriteDateFunctions(final String expression) {
        if (expression.contains("strftime_now"))
            return expression.replace("strftime_now", "jinjaStrftimeNow");
        return expression;
    }

    /** Escapes physical newlines found inside quoted string literals. */
    private String escapeNewlinesInStringLiterals(final String expression) {
        final var escaped = new StringBuilder(expression.length());
        char quote = 0;
        var escapedCharacter = false;
        for (var index = 0; index < expression.length(); index++) {
            final var current = expression.charAt(index);
            if (quote == 0) {
                escaped.append(current);
                if (current == '\'' || current == '"')
                    quote = current;
                continue;
            }
            if (current == '\r' || current == '\n') {
                if (current == '\r' && index + 1 < expression.length() && expression.charAt(index + 1) == '\n')
                    index++;
                escaped.append("\\n");
                escapedCharacter = false;
                continue;
            }
            escaped.append(current);
            if (escapedCharacter)
                escapedCharacter = false;
            else if (current == '\\')
                escapedCharacter = true;
            else if (current == quote)
                quote = 0;
        }
        return escaped.toString();
    }

    /** Rewrites a top-level Jinja inline conditional as a Groovy ternary. */
    private String rewriteInlineConditional(final String expression) {
        final var grouped = rewriteInlineConditionalsInGroups(expression);
        final var ifPosition = JinjaSourceScanning.findTopLevel(grouped, " if ", 0);
        if (ifPosition < 0)
            return grouped;

        final var elsePosition = JinjaSourceScanning.findTopLevel(grouped, " else ", ifPosition + 4);
        if (elsePosition < 0)
            throw new IllegalArgumentException("Inline Jinja conditional has no else branch: " + expression);

        final var whenTrue = grouped.substring(0, ifPosition);
        final var condition = grouped.substring(ifPosition + 4, elsePosition);
        final var whenFalse = grouped.substring(elsePosition + 6);

        return "((" + rewrite(condition, null) + ") ? ("
                + rewrite(whenTrue, null) + ") : (" + rewrite(whenFalse, null) + "))";
    }

    /** Recursively rewrites inline conditionals inside parenthesized groups. */
    private String rewriteInlineConditionalsInGroups(final String expression) {
        final var rewritten = new StringBuilder(expression.length());
        var position = 0;
        char quote = 0;
        var escaped = false;

        while (position < expression.length()) {
            final var current = expression.charAt(position);
            if (quote != 0) {
                rewritten.append(current);
                if (escaped) escaped = false;
                else if (current == '\\') escaped = true;
                else if (current == quote) quote = 0;
                position++;
                continue;
            }
            if (current == '\'' || current == '"') {
                rewritten.append(current);
                quote = current;
                position++;
                continue;
            }
            if (current != '(') {
                rewritten.append(current);
                position++;
                continue;
            }
            final var closing = findMatchingParenthesis(expression, position);
            if (closing < 0)
                throw new IllegalArgumentException("Unclosed parenthesis in expression: " + expression);
            rewritten.append('(')
                    .append(rewriteInlineConditional(expression.substring(position + 1, closing)))
                    .append(')');
            position = closing + 1;
        }

        return rewritten.toString();
    }

    /** Finds the closing parenthesis matching the supplied opening index. */
    private int findMatchingParenthesis(final String expression, final int opening) {
        var depth = 0;
        char quote = 0;
        var escaped = false;
        for (var index = opening; index < expression.length(); index++) {
            final var current = expression.charAt(index);
            if (quote != 0) {
                if (escaped)
                    escaped = false;
                else if (current == '\\')
                    escaped = true;
                else if (current == quote)
                    quote = 0;
                continue;
            }
            if (current == '\'' || current == '"')
                quote = current;
            else if (current == '(')
                depth++;
            else if (current == ')' && --depth == 0)
                return index;
        }

        return -1;
    }

    /** Rewrites supported Jinja filters as calls to generated helper closures. */
    private String rewriteFilters(final String source) {
        final var expression = new StringBuilder(source);
        var searchFrom = 0;
        while (true) {
            final var pipe = JinjaSourceScanning.findPipeOutsideString(expression, searchFrom);
            if (pipe < 0)
                return expression.toString();

            final var filterStart = skipWhitespace(expression, pipe + 1);
            final var filterEnd = consumeIdentifier(expression, filterStart);
            if (filterStart == filterEnd)
                throw new IllegalArgumentException("Expected a filter name after | in: " + source);

            final var operandStart = findFilterOperandStart(expression, pipe);
            if (operandStart == pipe)
                throw new IllegalArgumentException("Expected a filter operand before | in: " + source);

            final var operand = expression.substring(operandStart, pipe).trim();
            final var filter = expression.substring(filterStart, filterEnd);
            final var helper = switch (filter) {
                case "length" -> "jinjaLength";
                case "first" -> "jinjaFirst";
                case "last" -> "jinjaLast";
                case "tojson" -> "jinjaToJson";
                case "trim" -> "jinjaTrim";
                case "string" -> "jinjaStringValue";
                default -> throw new IllegalArgumentException("Unsupported Jinja filter: " + filter);
            };
            final var replacement = helper + "(" + operand + ")";
            expression.replace(operandStart, filterEnd, replacement);
            searchFrom = operandStart + replacement.length();
        }
    }

    /** Rewrites quoted-string containment expressions to Groovy string calls. */
    private String rewriteStringContains(final String expression) {
        return expression.replaceAll(
                "(\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*')\\s+in\\s+([A-Za-z_][A-Za-z0-9_.]*)",
                "$2.contains($1)");
    }


    /** Finds the start of the operand immediately preceding a filter pipe. */
    private int findFilterOperandStart(final StringBuilder expression, final int pipe) {
        var depth = 0;
        var index = pipe - 1;
        while (index >= 0 && Character.isWhitespace(expression.charAt(index))) index--;
        for (; index >= 0; index--) {
            final var current = expression.charAt(index);
            if (current == ')' || current == ']') depth++;
            else if (current == '(' || current == '[') {
                if (depth == 0) return index + 1;
                depth--;
            } else if (depth == 0 && isExpressionBoundary(current)) return index + 1;
        }
        return 0;
    }

    /** Determines whether a character separates expression operands. */
    private boolean isExpressionBoundary(final char value) {
        return Character.isWhitespace(value) || ",:+|&!>=/<*-".indexOf(value) >= 0;
    }

    /** Rewrites Jinja slice syntax as calls to the generated slice helper. */
    private String rewriteSlices(final String expression) {
        final var matcher = SLICE.matcher(expression);
        final var result = new StringBuilder();

        while (matcher.find())
            matcher.appendReplacement(result, Matcher.quoteReplacement(formatSlice(matcher)));

        matcher.appendTail(result);

        return result.toString();
    }

    /** Formats a matched slice expression as a helper invocation. */
    private String formatSlice(final Matcher matcher) {
        return "jinjaSlice(%s, %s, %s, %s)".formatted(
                matcher.group(1), nullable(matcher.group(2)), nullable(matcher.group(3)), nullable(matcher.group(4)));
    }

    /** Converts an omitted slice component into the generated {@code null} value. */
    private String nullable(final String value) {
        return value == null || value.isEmpty() ? "null" : value;
    }

    /** Converts supported Python/Jinja string method names to Groovy names. */
    private String rewriteStringMethods(final String expression) {
        return expression.replace(".lstrip('\\n')", ".stripLeading()")
                .replace(".rstrip('\\n')", ".stripTrailing()")
                .replace(".lstrip()", ".stripLeading()")
                .replace(".rstrip()", ".stripTrailing()")
                .replace(".strip('\\n')", ".strip()")
                .replace(".startswith(", ".startsWith(")
                .replace(".endswith(", ".endsWith(");
    }

    /** Rewrites operators while preserving text inside string literals. */
    private String rewriteOperators(final String expression) {
        final var rewritten = new StringBuilder(expression.length());
        final var outsideString = new StringBuilder();
        char quote = 0;
        var escaped = false;

        for (var index = 0; index < expression.length(); index++) {
            final var current = expression.charAt(index);
            if (quote != 0) {
                rewritten.append(current);
                if (escaped)
                    escaped = false;
                else if (current == '\\')
                    escaped = true;
                else if (current == quote)
                    quote = 0;
            } else if (current == '\'' || current == '"') {
                rewritten.append(rewriteOperatorsOutsideStrings(outsideString));
                outsideString.setLength(0);
                rewritten.append(current);
                quote = current;
            } else {
                outsideString.append(current);
            }
        }
        rewritten.append(rewriteOperatorsOutsideStrings(outsideString));

        return rewritten.toString();
    }

    /** Rewrites operators in a source fragment known to be outside strings. */
    private String rewriteOperatorsOutsideStrings(final StringBuilder expression) {
        return expression.toString().replaceAll("\\bis not none\\b", " != null")
                .replaceAll("\\bis none\\b", " == null")
                .replaceAll("\\b([A-Za-z_][A-Za-z0-9_]*)\\s+is\\s+not\\s+defined\\b", "!getBinding().hasVariable(\"$1\")")
                .replaceAll("\\b([A-Za-z_][A-Za-z0-9_]*)\\s+is\\s+defined\\b", "getBinding().hasVariable(\"$1\")")
                .replaceAll("\\bis string\\b", " instanceof CharSequence")
                .replaceAll("\\bis true\\b", " == true")
                .replaceAll("\\bis false\\b", " == false")
                .replaceAll("\\bnot\\s*\\(", "!(")
                .replaceAll("\\bnot\\s+", "!")
                .replaceAll("\\band\\b", "&&")
                .replaceAll("\\bor\\b", "||")
                .replaceAll("\\bnone\\b", "null");
    }

    /** Replaces Jinja loop references with the generated loop metadata variable. */
    private String rewriteLoopReference(final String expression, final String loopVariable) {
        final var rewritten = new StringBuilder(expression.length());
        var position = 0;
        while (position < expression.length()) {
            final var reference = JinjaSourceScanning.findLoopReferenceOutsideString(expression, position);
            if (reference < 0) {
                rewritten.append(expression, position, expression.length());
                break;
            }
            rewritten.append(expression, position, reference).append(loopVariable);
            position = reference + 4;
        }
        return rewritten.toString();
    }


    /** Returns the first non-whitespace position at or after the supplied index. */
    private int skipWhitespace(final StringBuilder source, final int from) {
        var position = from;
        while (position < source.length() && Character.isWhitespace(source.charAt(position)))
            position++;

        return position;
    }

    /** Consumes an identifier and returns the position immediately after it. */
    private int consumeIdentifier(final StringBuilder source, final int from) {
        var position = from;
        while (position < source.length() && JinjaSourceScanning.isIdentifierCharacter(source.charAt(position)))
            position++;

        return position;
    }
}
