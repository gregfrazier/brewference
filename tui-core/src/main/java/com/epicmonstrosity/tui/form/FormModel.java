package com.epicmonstrosity.tui.form;

import java.util.ArrayList;
import java.util.List;

/**
 * A form model managing field navigation, input, and validation.
 * @param <T> The type of the form data
 */
public final class FormModel<T> {
    private final Class<T> type;
    private final T original;
    private final List<FormFieldState> fields;
    private final List<FormConstraint> constraints = new ArrayList<>();
    private int cursor;
    private String status = "";

    /**
     * Create a form model.
     *
     * @param type The class type
     * @param original The original instance (can be null)
     */
    public FormModel(final Class<T> type, final T original) {
        this.type = type;
        this.original = original;
        this.fields = FormBinder.read(type, original);
    }

    /**
     * Add a custom constraint.
     *
     * @param constraint The constraint to add
     * @return This FormModel for chaining
     */
    public FormModel<T> constrain(final FormConstraint constraint) {
        if (constraint != null) {
            constraints.add(constraint);
        }
        return this;
    }

    /**
     * Get all form fields.
     *
     * @return List of fields
     */
    public List<FormFieldState> fields() {
        return fields;
    }

    /**
     * Get the cursor position.
     *
     * @return The cursor index
     */
    public int cursor() {
        return cursor;
    }

    /**
     * Get the status message.
     *
     * @return The status
     */
    public String status() {
        return status;
    }

    /**
     * Get the selected field.
     *
     * @return The selected field, or null if none
     */
    public FormFieldState selected() {
        if (fields.isEmpty()) {
            return null;
        }
        return fields.get(cursor);
    }

    /**
     * Move the cursor by a delta.
     *
     * @param delta The delta to move
     */
    public void moveBy(final int delta) {
        if (fields.isEmpty()) {
            return;
        }
        cursor = Math.max(0, Math.min(fields.size() - 1, cursor + delta));
    }

    /**
     * Type a character into the current field.
     *
     * @param text The character
     */
    public void type(final String text) {
        final FormFieldState field = selected();
        if (field == null || field.readOnly() || isToggleType(field.type())) {
            return;
        }
        field.setText(field.text() + (text == null ? "" : text));
        status = "";
    }

    /**
     * Paste text into the current field.
     *
     * @param text The text to paste
     */
    public void paste(final String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        String normalized = text.replace("\r\n", "\n").replace('\r', '\n');
        final FormFieldState field = selected();
        if (field != null && !field.multiline()) {
            normalized = normalized.replace("\n", "");
        }
        type(normalized);
    }

    /**
     * Add a newline to the current field (multiline only).
     */
    public void newline() {
        final FormFieldState field = selected();
        if (field == null || field.readOnly() || !field.multiline()) {
            return;
        }
        field.setText(field.text() + "\n");
        status = "";
    }

    /**
     * Delete the last character from the current field.
     */
    public void backspace() {
        final FormFieldState field = selected();
        if (field == null || field.readOnly() || isToggleType(field.type())) {
            return;
        }
        final String text = field.text();
        if (!text.isEmpty()) {
            field.setText(text.substring(0, text.length() - 1));
        }
        status = "";
    }

    /**
     * Activate the current field (toggle or cycle).
     */
    public void activate() {
        final FormFieldState field = selected();
        if (field == null || field.readOnly()) {
            return;
        }
        if (field.type() == boolean.class || field.type() == Boolean.class) {
            field.setValue(!Boolean.TRUE.equals(field.asBoolean()));
        } else if (field.type().isEnum()) {
            cycleEnum(field);
        }
        status = "";
    }

    /**
     * Submit the form and validate.
     *
     * @return The written instance if valid, null otherwise
     */
    public T submit() {
        final String error = FormBinder.validate(fields, constraints);
        if (error != null) {
            status = error;
            return null;
        }
        status = "Saved.";
        return FormBinder.write(type, original, fields);
    }

    /**
     * Cycle an enum field to the next value.
     *
     * @param field The field to cycle
     */
    private static void cycleEnum(final FormFieldState field) {
        final Object[] constants = field.type().getEnumConstants();
        if (constants == null || constants.length == 0) {
            return;
        }
        int index = 0;
        if (field.value() instanceof final Enum<?> current) {
            index = (current.ordinal() + 1) % constants.length;
        }
        field.setValue(constants[index]);
    }

    /**
     * Check if a type is a toggle type (boolean or enum).
     *
     * @param type The type to check
     * @return true if toggle type
     */
    private static boolean isToggleType(final Class<?> type) {
        return type == boolean.class || type == Boolean.class || type.isEnum();
    }
}
