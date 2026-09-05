package com.epicmonstrosity.tui.list;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * A file picker model for navigating the filesystem.
 */
public final class FilePickerModel implements Navigable<FilePickerModel.Entry> {
    public enum Mode { FILES, DIRECTORIES, ANY }

    /**
     * A file system entry.
     */
    public record Entry(Path path, Kind kind) {
        /** Parent directory (..). */
        public enum Kind { PARENT, DIRECTORY, FILE }

        /**
         * Get the display label for the entry.
         *
         * @return The label
         */
        public String label() {
            return switch (kind) {
                case PARENT -> "..";
                case DIRECTORY -> fileName(path) + "/";
                case FILE -> fileName(path);
            };
        }

        /**
         * Check if the entry is navigable.
         *
         * @return true if navigable
         */
        public boolean navigable() {
            return kind == Kind.PARENT || kind == Kind.DIRECTORY;
        }

        /**
         * Get the file name from a path.
         *
         * @param path The path
         * @return The file name
         */
        private static String fileName(final Path path) {
            final Path name = path.getFileName();
            return name == null ? path.toString() : name.toString();
        }
    }

    private final Mode mode;
    private final ListModel<Entry> items = new ListModel<>(List.of(), Entry::label);
    private Path current;
    private String error = "";

    private FilePickerModel(final Path start, final Mode mode) {
        this.mode = mode;
        this.current = start == null
                ? Path.of(".").toAbsolutePath().normalize()
                : start.toAbsolutePath().normalize();
        refresh();
    }

    /**
     * Create a file picker for files only.
     *
     * @param start The starting directory
     * @return The FilePickerModel
     */
    public static FilePickerModel files(final Path start) {
        return new FilePickerModel(start, Mode.FILES);
    }

    /**
     * Create a file picker for directories only.
     *
     * @param start The starting directory
     * @return The FilePickerModel
     */
    public static FilePickerModel directories(final Path start) {
        return new FilePickerModel(start, Mode.DIRECTORIES);
    }

    /**
     * Create a file picker for files and directories.
     *
     * @param start The starting directory
     * @return The FilePickerModel
     */
    public static FilePickerModel any(final Path start) {
        return new FilePickerModel(start, Mode.ANY);
    }

    /**
     * Get the current directory.
     *
     * @return The current path
     */
    public Path current() {
        return current;
    }

    /**
     * Get the picker mode.
     *
     * @return The mode
     */
    public Mode mode() {
        return mode;
    }

    /**
     * Get the error message.
     *
     * @return The error
     */
    public String error() {
        return error;
    }

    /**
     * Get all visible items.
     *
     * @return The visible items
     */
    public List<Entry> visibleItems() {
        return items.visibleItems();
    }

    /**
     * Get the visible window of items.
     *
     * @param viewportHeight The viewport height
     * @return The visible items
     */
    public List<Entry> visibleWindow(final int viewportHeight) {
        return items.visibleWindow(viewportHeight);
    }

    /**
     * Get the scroll offset.
     *
     * @return The scroll offset
     */
    public int scrollOffset() {
        return items.scrollOffset();
    }

    /**
     * Get the filter text.
     *
     * @return The filter
     */
    public String filter() {
        return items.filter();
    }

    /**
     * Set the filter text.
     *
     * @param filter The filter
     */
    public void setFilter(final String filter) {
        items.setFilter(filter);
    }

    /**
     * Type a character into the filter.
     *
     * @param text The character
     */
    public void typeFilter(final String text) {
        items.typeFilter(text);
    }

    /**
     * Delete the last character from the filter.
     */
    public void backspaceFilter() {
        items.backspaceFilter();
    }

    /**
     * Enter the selected entry if navigable.
     *
     * @return true if navigable
     */
    public boolean enterSelected() {
        return selectEntry()
                .filter(Entry::navigable)
                .map(entry -> {
                    current = entry.path().toAbsolutePath().normalize();
                    items.setFilter("");
                    refresh();
                    return true;
                })
                .orElse(false);
    }

    /**
     * Pick the selected entry.
     *
     * @return Optional of the picked path
     */
    public Optional<Path> pickSelected() {
        return selectEntry().flatMap(this::pickablePath);
    }

    /**
     * Pick the current directory.
     *
     * @return Optional of the current path
     */
    public Optional<Path> pickCurrentDirectory() {
        if (mode == Mode.FILES) {
            return Optional.empty();
        }
        return Optional.of(current);
    }

    /**
     * Refresh the directory listing.
     */
    public void refresh() {
        error = "";
        final List<Entry> entries = new ArrayList<>();
        final Path parent = current.getParent();
        if (parent != null) {
            entries.add(new Entry(parent, Entry.Kind.PARENT));
        }
        try (var stream = Files.list(current)) {
            final List<Path> children = stream.toList();
            children.stream()
                    .filter(Files::isDirectory)
                    .sorted(nameOrder())
                    .forEach(path -> entries.add(new Entry(path, Entry.Kind.DIRECTORY)));
            if (mode != Mode.DIRECTORIES) {
                children.stream()
                        .filter(path -> !Files.isDirectory(path))
                        .sorted(nameOrder())
                        .forEach(path -> entries.add(new Entry(path, Entry.Kind.FILE)));
            }
        } catch (IOException ex) {
            error = ex.getMessage() == null ? "Unable to list directory" : ex.getMessage();
        }
        items.setItems(entries);
        items.setCursor(0);
    }

    @Override
    public int getEntriesSize() {
        return items.getEntriesSize();
    }

    @Override
    public int getCursor() {
        return items.getCursor();
    }

    @Override
    public void setCursor(final int cursor) {
        items.setCursor(cursor);
    }

    @Override
    public Entry getEntry(final int index) {
        return items.getEntry(index);
    }

    /**
     * Get the pickable path for an entry.
     *
     * @param entry The entry
     * @return Optional of the path
     */
    private Optional<Path> pickablePath(final Entry entry) {
        return switch (entry.kind()) {
            case FILE -> mode == Mode.DIRECTORIES ? Optional.empty() : Optional.of(entry.path());
            case DIRECTORY -> mode == Mode.FILES ? Optional.empty() : Optional.of(entry.path());
            case PARENT -> Optional.empty();
        };
    }

    /**
     * Comparator for ordering entries by name.
     *
     * @return The comparator
     */
    private static Comparator<Path> nameOrder() {
        return Comparator.comparing(
                (Path path) -> {
                    final Path name = path.getFileName();
                    return name == null ? path.toString() : name.toString();
                },
                String.CASE_INSENSITIVE_ORDER
        );
    }
}
