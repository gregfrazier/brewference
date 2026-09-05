package com.epicmonstrosity.tui.form;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FormBinderTest {
    enum Role { USER, ADMIN }

    record SampleUser(
            @FormField(label = "Name", required = true, order = 1) String name,
            @FormField(label = "Age", order = 2) int age,
            @FormField(label = "Role", order = 3) Role role,
            @FormField(label = "Active", order = 4) boolean active
    ) {
    }

    static final class SampleBean {
        @FormField(label = "Name", required = true)
        private String name = "";
        @FormField(label = "Age")
        private int age;

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }
    }

    static final class NoSetterBean {
        @FormField(label = "Title")
        private final String title;

        NoSetterBean(String title) {
            this.title = title;
        }
    }

    @Test
    void recordRoundTrip() {
        SampleUser original = new SampleUser("Ada", 36, Role.ADMIN, true);
        List<FormFieldState> fields = FormBinder.read(SampleUser.class, original);
        assertEquals(4, fields.size());
        fields.get(0).setText("Grace");
        fields.get(1).setText("42");
        assertNull(FormBinder.validate(fields));
        SampleUser written = FormBinder.write(SampleUser.class, original, fields);
        assertEquals(new SampleUser("Grace", 42, Role.ADMIN, true), written);
    }

    @Test
    void pojoRoundTrip() {
        SampleBean bean = new SampleBean();
        bean.setName("Lin");
        bean.setAge(20);
        List<FormFieldState> fields = FormBinder.read(SampleBean.class, bean);
        fields.getFirst().setText("Kay");
        fields.get(1).setText("21");
        assertNull(FormBinder.validate(fields));
        SampleBean written = FormBinder.write(SampleBean.class, bean, fields);
        assertEquals("Kay", written.getName());
        assertEquals(21, written.getAge());
    }

    @Test
    void requiredBlankFieldBlocksSubmit() {
        List<FormFieldState> fields = FormBinder.read(SampleUser.class, new SampleUser("", 1, Role.USER, false));
        fields.getFirst().setText("   ");
        String error = FormBinder.validate(fields);
        assertNotNull(error);
        assertTrue(error.toLowerCase().contains("required"));
        assertEquals("Required", fields.getFirst().error());
    }

    @Test
    void numericParseErrorStaysOnField() {
        List<FormFieldState> fields = FormBinder.read(SampleUser.class, new SampleUser("Ada", 1, Role.USER, false));
        fields.get(1).setText("nope");
        String error = FormBinder.validate(fields);
        assertNotNull(error);
        assertEquals("Enter a valid number", fields.get(1).error());
    }

    @Test
    void missingSetterFailsFastWhenNoWritableField() {
        assertThrows(IllegalStateException.class, () ->
                FormBinder.write(NoSetterBean.class, new NoSetterBean("x"),
                        FormBinder.read(NoSetterBean.class, new NoSetterBean("x"))));
    }

    record ExtraUser(
            @FormField(label = "Email", required = true, pattern = "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$") String email,
            @FormField(label = "Age", min = "18", max = "120") int age,
            @FormField(label = "Home") Path home,
            @FormField(label = "Started") LocalDate started,
            @FormField(label = "Seen") Instant seen,
            @FormField(label = "Secret", masked = true) String secret,
            @FormField(label = "Notes", multiline = true) String notes
    ) {
    }

    @Test
    void pathDateAndInstantRoundTrip() {
        ExtraUser original = new ExtraUser(
                "ada@example.com",
                36,
                Path.of("docs"),
                LocalDate.of(2024, 1, 2),
                Instant.parse("2024-01-02T03:04:05Z"),
                "hunter2",
                "line1\nline2"
        );
        List<FormFieldState> fields = FormBinder.read(ExtraUser.class, original);
        assertEquals(7, fields.size());
        assertTrue(fields.get(5).masked());
        assertTrue(fields.get(6).multiline());
        fields.get(2).setText("home/ada");
        fields.get(3).setText("2026-08-17");
        fields.get(4).setText("2026-08-17T13:24:00Z");
        assertNull(FormBinder.validate(fields));
        ExtraUser written = FormBinder.write(ExtraUser.class, original, fields);
        assertEquals(Path.of("home/ada"), written.home());
        assertEquals(LocalDate.of(2026, 8, 17), written.started());
        assertEquals(Instant.parse("2026-08-17T13:24:00Z"), written.seen());
        assertEquals("hunter2", written.secret());
    }

    @Test
    void patternAndMinMaxStayOnField() {
        ExtraUser original = new ExtraUser("ada@example.com", 36, Path.of("."), LocalDate.now(), Instant.now(), "x", "");
        List<FormFieldState> fields = FormBinder.read(ExtraUser.class, original);
        fields.get(0).setText("not-an-email");
        assertEquals("Must match pattern", FormBinder.validate(fields));
        assertEquals("Must match pattern", fields.get(0).error());

        fields.get(0).setText("ada@example.com");
        fields.get(1).setText("12");
        assertTrue(FormBinder.validate(fields).toLowerCase().contains("min"));
        assertNotNull(fields.get(1).error());

        fields.get(1).setText("200");
        assertTrue(FormBinder.validate(fields).toLowerCase().contains("max"));
    }

    @Test
    void customConstraintBlocksSubmit() {
        List<FormFieldState> fields = FormBinder.read(SampleUser.class, new SampleUser("Ada", 36, Role.USER, true));
        String error = FormBinder.validate(fields, List.of(field ->
                "name".equals(field.name()) && "Ada".equals(field.text()) ? "Taken" : null));
        assertEquals("Taken", error);
        assertEquals("Taken", fields.getFirst().error());
    }
}
