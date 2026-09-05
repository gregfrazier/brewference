package com.epicmonstrosity.tui.form;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Bind form fields to POJOs and records, validate, and write back.
 */
public final class FormBinder {
    private FormBinder() { }

    /**
     * Read form field states from a POJO or record.
     *
     * @param type The class type
     * @param instance The instance to read from (can be null)
     * @return List of FormFieldState objects
     */
    public static <T> List<FormFieldState> read(final Class<T> type, final T instance) {
        if (type.isRecord()) {
            return readRecord(type, instance);
        }
        return readPojo(type, instance);
    }

    /**
     * Write form field states back to a POJO or record.
     *
     * @param type The class type
     * @param fields The list of field states to write
     * @return The written instance
     */
    public static <T> T write(final Class<T> type, final List<FormFieldState> fields) {
        return write(type, null, fields);
    }

    /**
     * Write form field states back to an instance.
     *
     * @param type The class type
     * @param original The original instance (can be null)
     * @param fields The list of field states to write
     * @return The written instance
     */
    @SuppressWarnings("unchecked")
    public static <T> T write(final Class<T> type, final T original, final List<FormFieldState> fields) {
        final Map<String, FormFieldState> byName = index(fields);
        if (type.isRecord()) {
            return (T) writeRecord(type, original, byName);
        }
        return writePojo(type, original, byName);
    }

    /**
     * Validate form fields with built-in constraints.
     *
     * @param fields The list of field states
     * @return Error message, or null if valid
     */
    public static String validate(final List<FormFieldState> fields) {
        return validate(fields, List.of());
    }

    /**
     * Validate form fields with built-in and custom constraints.
     *
     * @param fields The list of field states
     * @param constraints Custom constraints to apply
     * @return Error message, or null if valid
     */
    public static String validate(final List<FormFieldState> fields, final List<FormConstraint> constraints) {
        for (final FormFieldState field : fields) {
            field.setError(null);
            if (field.readOnly()) {
                continue;
            }
            try {
                final Object parsed = parseValue(field);
                if (field.required() && isBlank(parsed)) {
                    return fail(field, "Required");
                }
                field.setValue(parsed);
                final String rangeError = rangeError(field, parsed);
                if (rangeError != null) {
                    return fail(field, rangeError);
                }
                final String patternError = patternError(field);
                if (patternError != null) {
                    return fail(field, patternError);
                }
            } catch (final IllegalArgumentException ex) {
                return fail(field, ex.getMessage());
            }
        }
        if (constraints != null) {
            for (final FormConstraint constraint : constraints) {
                if (constraint == null) {
                    continue;
                }
                for (final FormFieldState field : fields) {
                    if (field.readOnly()) {
                        continue;
                    }
                    final String extra = constraint.validate(field);
                    if (extra != null && !extra.isBlank()) {
                        return fail(field, extra);
                    }
                }
            }
        }
        return null;
    }

    /**
     * Parse a field's text into its typed value.
     *
     * @param field The field state
     * @return The parsed value
     * @throws IllegalArgumentException If parsing fails
     */
    static Object parseValue(final FormFieldState field) {
        final Class<?> type = unwrap(field.type());
        final String text = field.text() == null ? "" : field.text();
        final String trimmed = text.trim();
        if (type == String.class) {
            return field.multiline() ? text : text;
        }
        if (type == boolean.class || type == Boolean.class) {
            return field.asBoolean();
        }
        if (type.isEnum()) {
            if (trimmed.isEmpty()) {
                return field.value();
            }
            try {
                return Enum.valueOf(type.asSubclass(Enum.class), trimmed);
            } catch (final IllegalArgumentException ex) {
                throw new IllegalArgumentException("Enter a valid value");
            }
        }
        if (type == Path.class) {
            if (trimmed.isEmpty()) {
                return null;
            }
            return Path.of(trimmed);
        }
        if (type == LocalDate.class) {
            if (trimmed.isEmpty()) {
                return null;
            }
            try {
                return LocalDate.parse(trimmed);
            } catch (final DateTimeParseException ex) {
                throw new IllegalArgumentException("Use YYYY-MM-DD");
            }
        }
        if (type == Instant.class) {
            if (trimmed.isEmpty()) {
                return null;
            }
            try {
                return Instant.parse(trimmed);
            } catch (final DateTimeParseException ex) {
                throw new IllegalArgumentException("Use ISO-8601 instant");
            }
        }
        if (isNumeric(type)) {
            if (trimmed.isEmpty()) {
                if (field.type().isPrimitive()) {
                    throw new IllegalArgumentException("Enter a number");
                }
                return null;
            }
            try {
                return parseNumber(type, trimmed);
            } catch (final NumberFormatException ex) {
                throw new IllegalArgumentException("Enter a valid number");
            }
        }
        throw new IllegalArgumentException("Unsupported field type: " + field.type().getName());
    }

    /**
     * Check range constraints on a parsed value.
     *
     * @param field The field state
     * @param parsed The parsed value
     * @return Error message, or null if valid
     */
    private static String rangeError(final FormFieldState field, final Object parsed) {
        if (parsed == null) {
            return null;
        }
        if (isNumeric(unwrap(field.type()))) {
            final double value = ((Number) parsed).doubleValue();
            if (!field.min().isBlank() && value < Double.parseDouble(field.min())) {
                return "Min " + field.min();
            }
            if (!field.max().isBlank() && value > Double.parseDouble(field.max())) {
                return "Max " + field.max();
            }
            return null;
        }
        if (parsed instanceof final String s) {
            if (!field.min().isBlank() && s.length() < Integer.parseInt(field.min())) {
                return "Min length " + field.min();
            }
            if (!field.max().isBlank() && s.length() > Integer.parseInt(field.max())) {
                return "Max length " + field.max();
            }
            return null;
        }
        if (parsed instanceof final LocalDate date) {
            if (!field.min().isBlank() && date.isBefore(LocalDate.parse(field.min()))) {
                return "Min " + field.min();
            }
            if (!field.max().isBlank() && date.isAfter(LocalDate.parse(field.max()))) {
                return "Max " + field.max();
            }
            return null;
        }
        if (parsed instanceof final Instant instant) {
            if (!field.min().isBlank() && instant.isBefore(Instant.parse(field.min()))) {
                return "Min " + field.min();
            }
            if (!field.max().isBlank() && instant.isAfter(Instant.parse(field.max()))) {
                return "Max " + field.max();
            }
        }
        return null;
    }

    /**
     * Check pattern constraint on a field.
     *
     * @param field The field state
     * @return Error message, or null if valid
     */
    private static String patternError(final FormFieldState field) {
        if (field.pattern().isBlank()) {
            return null;
        }
        final String text = field.text() == null ? "" : field.text();
        if (text.isEmpty() && !field.required()) {
            return null;
        }
        try {
            if (!Pattern.compile(field.pattern()).matcher(text).matches()) {
                return "Must match pattern";
            }
        } catch (final PatternSyntaxException ex) {
            return "Invalid pattern";
        }
        return null;
    }

    /**
     * Fail a field with an error message.
     *
     * @param field The field to fail
     * @param message The error message
     * @return The formatted error message
     */
    private static String fail(final FormFieldState field, final String message) {
        field.setError(message);
        return "Required".equals(message) ? field.label() + " is required" : message;
    }

    /**
     * Read form field states from a record.
     *
     * @param type The record class
     * @param instance The record instance (can be null)
     * @return List of FormFieldState objects
     */
    private static List<FormFieldState> readRecord(final Class<?> type, final Object instance) {
        final List<FormFieldState> fields = new ArrayList<>();
        int index = 0;
        for (final RecordComponent component : type.getRecordComponents()) {
            final FormField ann = component.getAnnotation(FormField.class);
            if (ann == null) {
                continue;
            }
            assertSupported(component.getType());
            Object value = defaultValue(component.getType());
            if (instance != null) {
                try {
                    value = component.getAccessor().invoke(instance);
                } catch (final ReflectiveOperationException ex) {
                    throw new IllegalStateException("Cannot read record component " + component.getName(), ex);
                }
            }
            fields.add(state(component.getName(), component.getType(), ann, value, index++));
        }
        fields.sort(Comparator.comparingInt(FormFieldState::order));
        return fields;
    }

    /**
     * Write form field states back to a record.
     *
     * @param type The record class
     * @param original The original record (can be null)
     * @param byName Map of field names to states
     * @return The written record
     */
    private static Object writeRecord(final Class<?> type, final Object original, final Map<String, FormFieldState> byName) {
        final RecordComponent[] components = type.getRecordComponents();
        final Class<?>[] types = new Class<?>[components.length];
        final Object[] args = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            types[i] = components[i].getType();
            final FormFieldState edited = byName.get(components[i].getName());
            if (edited != null) {
                args[i] = coerce(components[i].getType(), edited.value());
            } else if (original != null) {
                try {
                    args[i] = components[i].getAccessor().invoke(original);
                } catch (final ReflectiveOperationException ex) {
                    throw new IllegalStateException("Cannot read record component " + components[i].getName(), ex);
                }
            } else {
                args[i] = defaultValue(components[i].getType());
            }
        }
        try {
            final Constructor<?> ctor = type.getDeclaredConstructor(types);
            ctor.setAccessible(true);
            return ctor.newInstance(args);
        } catch (final ReflectiveOperationException ex) {
            throw new IllegalStateException("Cannot write record " + type.getName(), ex);
        }
    }

    /**
     * Read form field states from a POJO.
     *
     * @param type The POJO class
     * @param instance The POJO instance (can be null)
     * @return List of FormFieldState objects
     */
    private static List<FormFieldState> readPojo(final Class<?> type, final Object instance) {
        final List<Meta> metas = discoverPojo(type);
        final List<FormFieldState> fields = new ArrayList<>();
        int index = 0;
        for (final Meta meta : metas) {
            Object value = defaultValue(meta.type);
            if (instance != null) {
                value = readMember(instance, meta);
            }
            fields.add(state(meta.name, meta.type, meta.ann, value, index++));
        }
        fields.sort(Comparator.comparingInt(FormFieldState::order));
        return fields;
    }

    /**
     * Write form field states back to a POJO.
     *
     * @param type The POJO class
     * @param original The original POJO (can be null)
     * @param byName Map of field names to states
     * @return The written POJO
     */
    private static <T> T writePojo(final Class<T> type, final T original, final Map<String, FormFieldState> byName) {
        try {
            final T target = original != null ? original : type.getDeclaredConstructor().newInstance();
            for (final Meta meta : discoverPojo(type)) {
                final FormFieldState edited = byName.get(meta.name);
                if (edited == null || edited.readOnly()) {
                    continue;
                }
                writeMember(target, meta, coerce(meta.type, edited.value()));
            }
            return target;
        } catch (final NoSuchMethodException ex) {
            throw new IllegalStateException("No no-arg constructor or setters for " + type.getName(), ex);
        } catch (final ReflectiveOperationException ex) {
            throw new IllegalStateException("Cannot write " + type.getName(), ex);
        }
    }

    /**
     * Discover form fields from a POJO's fields and methods.
     *
     * @param type The POJO class
     * @return List of Meta objects describing form fields
     */
    private static List<Meta> discoverPojo(final Class<?> type) {
        final List<Meta> metas = new ArrayList<>();
        for (final Field field : type.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            final FormField ann = field.getAnnotation(FormField.class);
            if (ann == null) {
                continue;
            }
            assertSupported(field.getType());
            metas.add(new Meta(field.getName(), field.getType(), ann, field, null, setter(type, field)));
        }
        for (final Method method : type.getDeclaredMethods()) {
            final FormField ann = method.getAnnotation(FormField.class);
            if (ann == null || method.getParameterCount() != 0) {
                continue;
            }
            final String name = propertyName(method.getName());
            if (metas.stream().anyMatch(m -> m.name.equals(name))) {
                continue;
            }
            assertSupported(method.getReturnType());
            metas.add(new Meta(name, method.getReturnType(), ann, null, method, setter(type, name, method.getReturnType())));
        }
        if (metas.isEmpty()) {
            throw new IllegalStateException("No @FormField members on " + type.getName());
        }
        return metas;
    }

    /**
     * Read a member value via getter or field.
     *
     * @param instance The instance
     * @param meta The meta info
     * @return The member value
     */
    private static Object readMember(final Object instance, final Meta meta) {
        try {
            if (meta.getter != null) {
                meta.getter.setAccessible(true);
                return meta.getter.invoke(instance);
            }
            meta.field.setAccessible(true);
            return meta.field.get(instance);
        } catch (final ReflectiveOperationException ex) {
            throw new IllegalStateException("Cannot read " + meta.name, ex);
        }
    }

    /**
     * Write a member value via setter or field.
     *
     * @param instance The instance
     * @param meta The meta info
     * @param value The value to write
     * @throws ReflectiveOperationException If write fails
     */
    private static void writeMember(final Object instance, final Meta meta, final Object value) throws ReflectiveOperationException {
        if (meta.setter != null) {
            meta.setter.setAccessible(true);
            meta.setter.invoke(instance, value);
            return;
        }
        if (meta.field != null && !Modifier.isFinal(meta.field.getModifiers())) {
            meta.field.setAccessible(true);
            meta.field.set(instance, value);
            return;
        }
        throw new IllegalStateException("No setter or writable field for " + meta.name);
    }

    /**
     * Find a setter method for a field.
     *
     * @param type The class type
     * @param field The field
     * @return The setter method, or null
     */
    private static Method setter(final Class<?> type, final Field field) {
        return setter(type, field.getName(), field.getType());
    }

    /**
     * Find a setter method by name and value type.
     *
     * @param type The class type
     * @param name The field name
     * @param valueType The value type
     * @return The setter method, or null
     */
    private static Method setter(final Class<?> type, final String name, final Class<?> valueType) {
        final String setterName = "set" + Character.toUpperCase(name.charAt(0)) + name.substring(1);
        try {
            return type.getMethod(setterName, valueType);
        } catch (final NoSuchMethodException ignored) {
            try {
                return type.getDeclaredMethod(setterName, valueType);
            } catch (final NoSuchMethodException ex) {
                return null;
            }
        }
    }

    /**
     * Convert a method name to a property name.
     *
     * @param methodName The method name
     * @return The property name
     */
    private static String propertyName(final String methodName) {
        if (methodName.startsWith("get") && methodName.length() > 3) {
            return Character.toLowerCase(methodName.charAt(3)) + methodName.substring(4);
        }
        if (methodName.startsWith("is") && methodName.length() > 2) {
            return Character.toLowerCase(methodName.charAt(2)) + methodName.substring(3);
        }
        return methodName;
    }

    /**
     * Create a FormFieldState from a record component or POJO field.
     *
     * @param name The field name
     * @param type The field type
     * @param ann The FormField annotation
     * @param value The field value
     * @param index The order index
     * @return The FormFieldState
     */
    private static FormFieldState state(final String name, final Class<?> type, final FormField ann, final Object value, final int index) {
        final String label = ann.label().isBlank() ? name : ann.label();
        final int order = ann.order() == 0 ? index : ann.order();
        return new FormFieldState(
                name, type, label, ann.required(), ann.readOnly(), ann.help(), order, value,
                ann.masked(), ann.multiline(), ann.pattern(), ann.min(), ann.max()
        );
    }

    /**
     * Index form fields by name.
     *
     * @param fields The list of fields
     * @return Map of name to field
     */
    private static Map<String, FormFieldState> index(final List<FormFieldState> fields) {
        final Map<String, FormFieldState> map = new HashMap<>();
        for (final FormFieldState field : fields) {
            map.put(field.name(), field);
        }
        return map;
    }

    /**
     * Assert that a type is supported for form fields.
     *
     * @param type The type to check
     * @throws IllegalStateException If unsupported
     */
    private static void assertSupported(final Class<?> type) {
        final Class<?> raw = unwrap(type);
        if (raw == String.class || raw == boolean.class || raw == Boolean.class || raw.isEnum() || isNumeric(raw)
                || raw == Path.class || raw == LocalDate.class || raw == Instant.class) {
            return;
        }
        throw new IllegalStateException("Unsupported form field type: " + type.getName());
    }

    /**
     * Check if a type is numeric.
     *
     * @param type The type to check
     * @return true if numeric
     */
    private static boolean isNumeric(final Class<?> type) {
        return type == int.class || type == Integer.class
                || type == long.class || type == Long.class
                || type == double.class || type == Double.class
                || type == float.class || type == Float.class
                || type == short.class || type == Short.class
                || type == byte.class || type == Byte.class;
    }

    /**
     * Parse text as a number of the given type.
     *
     * @param type The numeric type
     * @param text The text to parse
     * @return The parsed number
     * @throws NumberFormatException If parsing fails
     */
    private static Object parseNumber(final Class<?> type, final String text) {
        if (type == int.class || type == Integer.class) return Integer.valueOf(text);
        if (type == long.class || type == Long.class) return Long.valueOf(text);
        if (type == double.class || type == Double.class) return Double.valueOf(text);
        if (type == float.class || type == Float.class) return Float.valueOf(text);
        if (type == short.class || type == Short.class) return Short.valueOf(text);
        if (type == byte.class || type == Byte.class) return Byte.valueOf(text);
        throw new NumberFormatException(text);
    }

    /**
     * Coerce a value to a type (for primitives).
     *
     * @param type The target type
     * @param value The value to coerce
     * @return The coerced value
     */
    private static Object coerce(final Class<?> type, final Object value) {
        if (value == null) {
            return type.isPrimitive() ? defaultValue(type) : null;
        }
        return value;
    }

    /**
     * Get the default value for a type.
     *
     * @param type The type
     * @return The default value
     */
    private static Object defaultValue(final Class<?> type) {
        if (!type.isPrimitive()) {
            return type == String.class ? "" : null;
        }
        if (type == boolean.class) return false;
        if (type == int.class || type == short.class || type == byte.class) return 0;
        if (type == long.class) return 0L;
        if (type == double.class) return 0.0d;
        if (type == float.class) return 0.0f;
        return null;
    }

    /**
     * Unwrap a type (identity).
     *
     * @param type The type
     * @return The type
     */
    private static Class<?> unwrap(final Class<?> type) {
        return type;
    }

    /**
     * Check if a value is blank.
     *
     * @param value The value to check
     * @return true if blank
     */
    private static boolean isBlank(final Object value) {
        return value == null || (value instanceof final String s && s.isBlank());
    }

    private record Meta(String name, Class<?> type, FormField ann, Field field, Method getter, Method setter) {
    }
}
