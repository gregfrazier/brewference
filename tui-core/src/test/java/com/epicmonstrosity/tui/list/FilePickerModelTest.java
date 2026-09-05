package com.epicmonstrosity.tui.list;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FilePickerModelTest {
    @TempDir
    Path temp;

    @Test
    void listsDirectoriesFirstThenFilesAndCanEnterChild() throws IOException {
        Files.createDirectory(temp.resolve("sub"));
        Files.writeString(temp.resolve("a.txt"), "a");
        Files.writeString(temp.resolve("sub").resolve("b.txt"), "b");

        FilePickerModel model = FilePickerModel.files(temp);
        List<String> names = model.visibleItems().stream().map(FilePickerModel.Entry::label).toList();
        assertTrue(names.contains(".."));
        assertEquals("sub/", names.get(names.indexOf("sub/")));
        assertTrue(names.indexOf("sub/") < names.indexOf("a.txt"));

        model.setCursor(indexOf(model, "sub/"));
        model.enterSelected();
        assertEquals(temp.resolve("sub").toAbsolutePath().normalize(), model.current());
        assertTrue(model.visibleItems().stream().anyMatch(entry -> "b.txt".equals(entry.label())));
    }

    @Test
    void directoriesModeHidesFiles() throws IOException {
        Files.createDirectory(temp.resolve("only-dir"));
        Files.writeString(temp.resolve("hidden.txt"), "x");

        FilePickerModel model = FilePickerModel.directories(temp);
        List<String> names = model.visibleItems().stream().map(FilePickerModel.Entry::label).toList();
        assertTrue(names.contains("only-dir/"));
        assertTrue(names.stream().noneMatch(name -> name.equals("hidden.txt")));
    }

    private static int indexOf(FilePickerModel model, String label) {
        List<FilePickerModel.Entry> items = model.visibleItems();
        for (int i = 0; i < items.size(); i++) {
            if (label.equals(items.get(i).label())) {
                return i;
            }
        }
        throw new AssertionError("missing " + label);
    }
}
