package com.epicmonstrosity.brewference.template.jinja;

import java.util.ArrayDeque;
import java.util.Deque;

/** Emits Groovy control-flow and assignment statements for Jinja tags. */
final class JinjaStatementWriter {
    private final Deque<String> loopVariables = new ArrayDeque<>();
    private final JinjaExpressionRewriter expressions;
    private int loopId;

    /** Creates a statement writer using the supplied expression rewriter. */
    JinjaStatementWriter(final JinjaExpressionRewriter expressions) {
        this.expressions = expressions;
    }

    /** Resets loop state before compiling a new template. */
    void reset() {
        loopVariables.clear();
        loopId = 0;
    }

    /** Returns the metadata variable for the currently open loop, if any. */
    String loopVariable() {
        return loopVariables.peek();
    }

    /** Reports whether an end tag is still required for an open loop. */
    boolean hasOpenLoop() {
        return !loopVariables.isEmpty();
    }

    /** Appends generated Groovy for one Jinja statement. */
    void append(final StringBuilder groovy, final String statement) {
        if (statement.isBlank() || statement.startsWith("#")) return;
        if (statement.startsWith("if ")) {
            appendCondition(groovy, "if (", statement.substring(3));
        } else if (statement.startsWith("elif ")) {
            appendCondition(groovy, "} else if (", statement.substring(5));
        } else if (statement.equals("else")) {
            groovy.append("} else {\n");
        } else if (statement.equals("endif")) {
            groovy.append("}\n");
        } else if (statement.startsWith("for ")) {
            appendFor(groovy, statement.substring(4));
        } else if (statement.equals("endfor")) {
            closeLoop(groovy);
        } else if (statement.startsWith("set ")) {
            appendSet(groovy, statement.substring(4));
        } else {
            throw new IllegalArgumentException("Unsupported Jinja statement: " + statement);
        }
    }

    /** Appends an if or else-if condition. */
    private void appendCondition(final StringBuilder groovy, final String prefix, final String condition) {
        groovy.append(prefix)
                .append(expressions.rewrite(condition, loopVariable()))
                .append(") {\n");
    }

    /** Appends a Groovy loop and records its loop metadata variable. */
    private void appendFor(final StringBuilder groovy, final String source) {
        final var inPosition = JinjaSourceScanning.findTopLevel(source, " in ", 0);
        if (inPosition < 1) throw new IllegalArgumentException("Invalid Jinja for-loop: " + source);
        final var variable = source.substring(0, inPosition).trim();
        final var collection = expressions.rewrite(source.substring(inPosition + 4), loopVariable());
        final var id = loopId++;
        final var collectionVariable = "__jinjaCollection" + id;
        final var indexVariable = "__jinjaIndex" + id;
        final var loopVariable = "__jinjaLoop" + id;
        groovy.append("def ").append(collectionVariable).append(" = ").append(collection).append('\n')
                .append("for (int ").append(indexVariable).append(" = 0; ")
                .append(indexVariable).append(" < ").append(collectionVariable).append(".size(); ")
                .append(indexVariable).append("++) {\n")
                .append("    def ").append(variable).append(" = ").append(collectionVariable)
                .append('[').append(indexVariable).append("]\n")
                .append("    def ").append(loopVariable).append(" = [index0: ").append(indexVariable)
                .append(", index: ").append(indexVariable).append(" + 1, first: ").append(indexVariable)
                .append(" == 0, last: ").append(indexVariable).append(" == ").append(collectionVariable)
                .append(".size() - 1]\n");
        loopVariables.push(loopVariable);
    }

    /** Closes the current loop or reports an unexpected end tag. */
    private void closeLoop(final StringBuilder groovy) {
        if (loopVariables.isEmpty()) throw new IllegalArgumentException("Unexpected Jinja endfor");
        loopVariables.pop();
        groovy.append("}\n");
    }

    /** Appends an assignment or namespace declaration. */
    private void appendSet(final StringBuilder groovy, final String source) {
        final var equalsPosition = JinjaSourceScanning.findTopLevel(source, "=", 0);
        if (equalsPosition < 1) throw new IllegalArgumentException("Invalid Jinja set statement: " + source);
        final var target = source.substring(0, equalsPosition).trim();
        final var value = source.substring(equalsPosition + 1).trim();
        if (value.startsWith("namespace(") && value.endsWith(")")) {
            appendNamespace(groovy, target, value);
        } else {
            groovy.append(target).append(" = ")
                    .append(expressions.rewrite(value, loopVariable())).append('\n');
        }
    }

    /** Appends creation and initialization of a Jinja namespace object. */
    private void appendNamespace(final StringBuilder groovy, final String target, final String expression) {
        groovy.append(target).append(" = new Expando()\n");
        final var arguments = expression.substring(10, expression.length() - 1);
        for (final var assignment : JinjaSourceScanning.splitTopLevel(arguments, ',')) {
            if (assignment.isBlank()) continue;
            final var equalsPosition = JinjaSourceScanning.findTopLevel(assignment, "=", 0);
            if (equalsPosition < 1) throw new IllegalArgumentException(
                    "Invalid namespace property assignment: " + assignment);
            groovy.append(target).append('.')
                    .append(assignment.substring(0, equalsPosition).trim()).append(" = ")
                    .append(expressions.rewrite(assignment.substring(equalsPosition + 1), loopVariable()))
                    .append('\n');
        }
    }

}
