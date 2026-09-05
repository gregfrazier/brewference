package com.epicmonstrosity.tui.list;

import com.epicmonstrosity.tui.KeyPressed;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TableModelTest {
    private record Row(String name, String kind, int size) {
    }

    private static final List<TableColumn<Row>> COLUMNS = List.of(
            TableColumn.of("Name", 12, Row::name)
                    .sortable(Comparator.comparing(Row::name, String.CASE_INSENSITIVE_ORDER)),
            TableColumn.of("Kind", 8, Row::kind),
            TableColumn.of("Size", 6, (Row row) -> Integer.toString(row.size()))
                    .sortable(Comparator.comparingInt(Row::size))
    );

    @Test
    void snapshotsAnyCollectionAsAList() {
        TableModel<Row> fromDeque = new TableModel<>(COLUMNS, new ArrayDeque<>(rows()));
        TableModel<Row> fromSet = new TableModel<>(COLUMNS, Set.of(new Row("solo", "a", 1)));
        assertEquals(3, fromDeque.size());
        assertEquals("solo", fromSet.items().getFirst().name());
    }

    @Test
    void addDoesNotReorderUntilSorted() {
        TableModel<Row> model = new TableModel<>(COLUMNS, List.of());
        model.add(new Row("c", "x", 3));
        model.add(new Row("a", "x", 1));
        assertEquals(List.of("c", "a"), names(model));
    }

    @Test
    void cycleSortSkipsUnsortableColumnsAndTogglesDirection() {
        TableModel<Row> model = new TableModel<>(COLUMNS, rows());
        model.setCursor(0);
        assertEquals("b", model.selectEntry().orElseThrow().name());

        model.cycleSort();
        assertEquals("Size", COLUMNS.get(model.sortColumn()).header());
        assertTrue(model.sortAscending());
        assertEquals(List.of("a", "b", "c"), names(model));
        assertEquals("b", model.selectEntry().orElseThrow().name());

        model.cycleSort();
        assertEquals("Name", COLUMNS.get(model.sortColumn()).header());
        assertFalse(model.sortAscending());
        assertEquals(List.of("c", "b", "a"), names(model));
        assertTrue(model.statusMessage().contains("Name"));
        assertTrue(model.statusMessage().contains("desc"));
    }

    @Test
    void addAfterSortKeepsOrder() {
        TableModel<Row> model = new TableModel<>(COLUMNS, rows());
        model.sortBy(0, true);
        model.add(new Row("aa", "z", 4));
        assertEquals(List.of("a", "aa", "b", "c"), names(model));
    }

    @Test
    void matchingSelectsEnqueueCandidates() {
        TableModel<Row> model = new TableModel<>(COLUMNS, rows());
        assertEquals(List.of("a"), model.matching(row -> row.size() < 2).stream().map(Row::name).toList());
        assertEquals(3, model.matching(null).size());
    }

    @Test
    void visibleWindowKeepsCursorInView() {
        TableModel<Row> model = new TableModel<>(COLUMNS, rows());
        model.addAll(List.of(new Row("d", "x", 4), new Row("e", "x", 5), new Row("f", "x", 6)));
        model.setCursor(4);
        assertEquals(List.of("c", "d", "e"), model.visibleWindow(3).stream().map(Row::name).toList());
    }

    @Test
    void clearResetsSortAndCursor() {
        TableModel<Row> model = new TableModel<>(COLUMNS, rows());
        model.cycleSort();
        model.setCursor(2);
        model.clear();
        assertEquals(0, model.size());
        assertFalse(model.sorted());
        assertEquals(0, model.getCursor());
    }

    @Test
    void screenEnqueuesSelectedAndMatchingRows() {
        List<String> queued = new ArrayList<>();
        TableScreen<Row> screen = new TableScreen<>("Rows", COLUMNS, rows())
                .onEnqueue(row -> queued.add(row.name()))
                .enqueueMatching(row -> row.size() >= 3);
        screen.onEvent(new KeyPressed("a"));
        screen.onEvent(new KeyPressed("p"));
        assertEquals(List.of("b", "c"), queued);
    }

    private static List<Row> rows() {
        return List.of(
                new Row("b", "beta", 2),
                new Row("a", "alpha", 1),
                new Row("c", "gamma", 3)
        );
    }

    private static List<String> names(final TableModel<Row> model) {
        return model.items().stream().map(Row::name).toList();
    }
}
