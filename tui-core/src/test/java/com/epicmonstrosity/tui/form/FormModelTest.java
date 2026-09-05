package com.epicmonstrosity.tui.form;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FormModelTest {
    record Person(
            @FormField(label = "Name") String name,
            @FormField(label = "Notes", multiline = true) String notes
    ) {
    }

    @Test
    void pasteStripsNewlinesOnSingleLineFields() {
        final FormModel<Person> model = new FormModel<>(Person.class, new Person("", ""));
        model.paste("Ada\r\nLovelace\n");
        assertEquals("AdaLovelace", model.selected().text());
    }

    @Test
    void pasteKeepsNewlinesOnMultilineFields() {
        final FormModel<Person> model = new FormModel<>(Person.class, new Person("Ada", "old"));
        model.moveBy(1);
        model.paste("line1\r\nline2");
        assertEquals("oldline1\nline2", model.selected().text());
    }
}
