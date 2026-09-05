package com.epicmonstrosity.tui.list;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ListModelTest {
    @Test
    void filterNarrowsVisibleItemsAndClampsCursor() {
        ListModel<String> model = new ListModel<>(List.of("Apple", "Apricot", "Banana", "Cherry"));
        model.setCursor(3);
        model.setFilter("ap");
        assertEquals(List.of("Apple", "Apricot"), model.visibleItems());
        assertEquals(0, model.getCursor());
    }

    @Test
    void multiSelectTracksItemsNotFilteredIndexes() {
        ListModel<String> model = new ListModel<>(List.of("Apple", "Banana", "Cherry"));
        model.toggleSelected();
        model.moveDown();
        model.moveDown();
        model.toggleSelected();
        assertEquals(List.of("Apple", "Cherry"), model.selectedItems());

        model.setFilter("ch");
        assertEquals(List.of("Cherry"), model.visibleItems());
        assertEquals(List.of("Apple", "Cherry"), model.selectedItems());
        assertTrue(model.isSelected(model.getEntry(0)));
    }

    @Test
    void visibleWindowKeepsCursorInView() {
        ListModel<String> model = new ListModel<>(List.of("a", "b", "c", "d", "e", "f"));
        model.setCursor(4);
        assertEquals(List.of("c", "d", "e"), model.visibleWindow(3));
    }
}
