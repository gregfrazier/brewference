package com.epicmonstrosity.tui.form;

/**
 * State of a form field (name, type, value, constraints).
 */
public final class FormFieldState {
    private final String name;
    private final Class<?> type;
    private final String label;
    private final boolean required;
    private final boolean readOnly;
    private final String help;
    private final int order;
    private final boolean masked;
    private final boolean multiline;
    private final String pattern;
    private final String min;
    private final String max;
    private Object value;
    private String text;
    private String error;

    /**
     * Create a form field state with default values.
     *
     * @param name The field name
     * @param type The field type
     * @param label The field label
     * @param required Whether the field is required
     * @param readOnly Whether the field is read-only
     * @param help Help text
     * @param value The initial value
     */
    public FormFieldState(final String name, final Class<?> type, final String label, final boolean required,
                          final boolean readOnly, final String help, final Object value) {
        this(name, type, label, required, readOnly, help, 0, value);
    }

    /**
     * Create a form field state with order.
     *
     * @param name The field name
     * @param type The field type
     * @param label The field label
     * @param required Whether the field is required
     * @param readOnly Whether the field is read-only
     * @param help Help text
     * @param order The field order
     * @param value The initial value
     */
    public FormFieldState(final String name, final Class<?> type, final String label, final boolean required,
                          final boolean readOnly, final String help, final int order, final Object value) {
        this(name, type, label, required, readOnly, help, order, value, false, false, "", "", "");
    }

    /**
     * Create a full form field state.
     *
     * @param name The field name
     * @param type The field type
     * @param label The field label
     * @param required Whether the field is required
     * @param readOnly Whether the field is read-only
     * @param help Help text
     * @param order The field order
     * @param value The initial value
     * @param masked Whether the field is password-masked
     * @param multiline Whether the field is multiline
     * @param pattern Regex pattern for validation
     * @param min Minimum value
     * @param max Maximum value
     */
    public FormFieldState(final String name, final Class<?> type, final String label, final boolean required,
                          final boolean readOnly, final String help, final int order, final Object value,
                          final boolean masked, final boolean multiline, final String pattern, final String min, final String max) {
        this.name = name;
        this.type = type;
        this.label = label;
        this.required = required;
        this.readOnly = readOnly;
        this.help = help;
        this.order = order;
        this.masked = masked;
        this.multiline = multiline;
        this.pattern = pattern == null ? "" : pattern;
        this.min = min == null ? "" : min;
        this.max = max == null ? "" : max;
        this.value = value;
        this.text = stringify(value);
    }

    /** Get the field name. */
    public String name() { return name; }
    /** Get the field type. */
    public Class<?> type() { return type; }
    /** Get the field label. */
    public String label() { return label; }
    /** Whether the field is required. */
    public boolean required() { return required; }
    /** Whether the field is read-only. */
    public boolean readOnly() { return readOnly; }
    /** Get the help text. */
    public String help() { return help; }
    /** The field order. */
    public int order() { return order; }
    /** Whether the field is masked. */
    public boolean masked() { return masked; }
    /** Whether the field is multiline. */
    public boolean multiline() { return multiline; }
    /** Get the pattern. */
    public String pattern() { return pattern; }
    /** Get the minimum value. */
    public String min() { return min; }
    /** Get the maximum value. */
    public String max() { return max; }
    /** Get the current value. */
    public Object value() { return value; }
    /** Get the text input. */
    public String text() { return text; }
    /** Get the error message. */
    public String error() { return error; }

    /**
     * Set the field value.
     *
     * @param value The new value
     */
    public void setValue(final Object value) {
        this.value = value;
        this.text = stringify(value);
        this.error = null;
    }

    /**
     * Set the text input.
     *
     * @param text The text
     */
    public void setText(final String text) {
        this.text = text == null ? "" : text;
        this.error = null;
    }

    /**
     * Set the error message.
     *
     * @param error The error
     */
    public void setError(final String error) {
        this.error = error;
    }

    /**
     * Get the display value for rendering.
     *
     * @return The display value
     */
    public String displayValue() {
        if (masked) {
            final int length = text == null ? 0 : text.length();
            return "*".repeat(length);
        }
        if (type.isEnum() && value instanceof final Enum<?> e) {
            return e.name();
        }
        if (type == boolean.class || type == Boolean.class) {
            return Boolean.toString(Boolean.TRUE.equals(asBoolean()));
        }
        final String raw = text == null ? "" : text;
        if (multiline) {
            return raw.replace("\n", "⏎");
        }
        return raw;
    }

    /**
     * Get the value as boolean.
     *
     * @return The boolean value
     */
    public Boolean asBoolean() {
        if (value instanceof final Boolean b) {
            return b;
        }
        return Boolean.parseBoolean(text);
    }

    /**
     * Convert an object to a string.
     *
     * @param value The object
     * @return The string representation
     */
    static String stringify(final Object value) {
        return value == null ? "" : String.valueOf(value);
    }
}
