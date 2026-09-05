package com.epicmonstrosity.brewference.template.jinja;

import com.epicmonstrosity.brewference.template.chat.ChatMessage;
import com.epicmonstrosity.brewference.template.chat.ChatRole;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Golden-file tests pinning the current Jinja-to-Groovy transpiler behavior. */
class JinjaGoldenTest {
    private final JinjaTemplateCompiler compiler = new JinjaTemplateCompiler();

    @Test
    void compilesGoldenTemplatesToPinnedGroovy() throws IOException {
        for (Path template : jinjaFiles()) {
            var name = template.getFileName().toString().replace(".jinja", "");
            var expected = Files.readString(goldenDirectory().resolve(name + ".groovy"));
            assertEquals(expected, compiler.compile(Files.readString(template)), name);
        }
    }

    @Test
    void rendersGoldenTemplatesThroughGroovyShell() throws IOException {
        for (Path template : jinjaFiles()) {
            var name = template.getFileName().toString().replace(".jinja", "");
            var expected = Files.readString(goldenDirectory().resolve(name + ".expected"));
            assertEquals(expected, render(Files.readString(template)), name);
        }
    }

    @Test
    void smolLm3() throws IOException {
        var template = Files.readString(modelJinjaDirectory().resolve("smollm3.jinja"));
        var expected = Files.readString(modelJinjaDirectory().resolve("smollm3.expected"));
        assertEquals(expected, compiler.compile(template));
    }

    @Test
    void rejectsNullTemplate() {
        assertMessage("jinja must not be null", () -> compiler.compile(null));
    }

    @Test
    void ignoresComments() {
        var groovy = compiler.compile("before{# comment #}after");

        assertFalse(groovy.contains("comment"));
        assertEquals(1, occurrences(groovy, "out << \"before\""));
        assertEquals(1, occurrences(groovy, "out << \"after\""));
    }

    @Test
    void supportsUndefinedVariableDefaults() {
        var groovy = compiler.compile("{%- if enable_thinking is not defined -%}"
                + "{%- set enable_thinking = true -%}{%- endif -%}");

        assertTrue(groovy.contains("if (!getBinding().hasVariable(\"enable_thinking\")) {"));
        assertTrue(groovy.contains("enable_thinking = true"));
    }

    @Test
    void missingVariablesAreSafeAndDefinedChecksTrackBindingPresence() {
        var template = "{% if xml_tools != null or python_tools != null %}present"
                + "{% else %}missing{% endif %}|"
                + "{% if xml_tools is defined %}defined{% else %}undefined{% endif %}";
        var renderer = new JinjaGroovyRenderer();

        assertEquals("missing|undefined", renderer.setPromptTemplate(template).render(List.of()));

        renderer.addBindingValue("xml_tools", null);
        assertEquals("missing|defined", renderer.render(List.of()));
    }

    @Test
    void rejectsUnterminatedTag() {
        assertMessage("Unterminated Jinja tag at offset 0", () -> compiler.compile("{{ name"));
    }

    @Test
    void rejectsUnclosedForLoop() {
        assertMessage("Unclosed Jinja for-loop", () -> compiler.compile("{% for item in items %}"));
    }

    @Test
    void rejectsUnexpectedEndfor() {
        assertMessage("Unexpected Jinja endfor", () -> compiler.compile("{% endfor %}"));
    }

    @Test
    void rejectsUnsupportedFilter() {
        assertMessage("Unsupported Jinja filter: bogus", () -> compiler.compile("{{ name | bogus }}"));
    }

    @Test
    void rejectsUnsupportedStatement() {
        assertMessage("Unsupported Jinja statement: include \"x\"", () -> compiler.compile("{% include \"x\" %}"));
    }

    @Test
    void rejectsInlineConditionalWithoutElse() {
        assertMessage("Inline Jinja conditional has no else branch: name if show",
                () -> compiler.compile("{{ name if show }}"));
    }

    /** Pins a known quirk: an inline conditional whose condition uses or crashes the filter pass. */
    @Test
    void rejectsInlineConditionalWithOrInCondition() {
        assertMessage("Expected a filter name after | in: ((missing || show) ? (\"a\") : (\"b\"))",
                () -> compiler.compile("{{ \"a\" if missing or show else \"b\" }}"));
    }

    private void assertMessage(String expected, Executable compilation) {
        var exception = assertThrows(IllegalArgumentException.class, compilation);
        assertEquals(expected, exception.getMessage());
    }

    private int occurrences(String source, String value) {
        return (source.length() - source.replace(value, "").length()) / value.length();
    }

    private List<Path> jinjaFiles() throws IOException {
        return Files.list(goldenDirectory()).filter(path -> path.toString().endsWith(".jinja")).sorted().toList();
    }

    private Path goldenDirectory() {
        try {
            return Path.of(JinjaGoldenTest.class.getResource("/jinja-golden").toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

    private Path modelJinjaDirectory() {
        try {
            return Path.of(JinjaGoldenTest.class.getResource("/jinja-models").toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Renders a template with the fixed binding set shared by all golden cases. */
    private String render(String template) {
        var renderer = new JinjaGroovyRenderer();
        renderer.addBindingValue("items", List.of("alpha", "beta", "gamma"));
        renderer.addBindingValue("padded", "  hello  ");
        renderer.addBindingValue("name", "world");
        renderer.addBindingValue("show", true);
        renderer.addBindingValue("missing", null);
        renderer.addBindingValue("line", "abc\n");
        renderer.addBindingValue("tools", List.of(Map.of("name", "get_weather")));
        return renderer.setPromptTemplate(template).render(List.of(
                new ChatMessage(ChatRole.SYSTEM, "You are helpful."),
                new ChatMessage(ChatRole.USER, "Hello"),
                new ChatMessage(ChatRole.MODEL, "Hi there")));
    }
}
