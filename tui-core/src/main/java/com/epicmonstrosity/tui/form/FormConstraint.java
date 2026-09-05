package com.epicmonstrosity.tui.form;

/**
 * A custom form field validation constraint.
 */
@FunctionalInterface
public interface FormConstraint {
    /**
     * Validate a field and return an error message.
     *
     * @param field The field to validate
     * @return An error message, or null if the field is acceptable
     */
    String validate(FormFieldState field);
}
