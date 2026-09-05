package com.epicmonstrosity.tui.form;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for form field metadata on fields, record components, or methods.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD, ElementType.RECORD_COMPONENT, ElementType.METHOD})
public @interface FormField {
    /** Field label. */
    String label() default "";
    /** Field order. */
    int order() default 0;
    /** Whether the field is required. */
    boolean required() default false;
    /** Whether the field is read-only. */
    boolean readOnly() default false;
    /** Help text shown for the field. */
    String help() default "";
    /** Whether the field is password-masked. */
    boolean masked() default false;
    /** Whether the field is multiline. */
    boolean multiline() default false;
    /** Regex pattern for validation. */
    String pattern() default "";
    /** Minimum value (numeric or length). */
    String min() default "";
    /** Maximum value (numeric or length). */
    String max() default "";
}
