# -*- coding: utf-8 -*-
"""Tests for LiveFrameSeries._IndexedList — the list-with-O(1)-membership
that lives behind ``LiveFrameSeries.index``.

Verifies that:
* All list reads behave identically to a plain list.
* Membership (``x in lst``) is O(1) via the parallel set.
* All mutating ops keep the set in sync.
* Equality + ordering are preserved.
"""

from __future__ import annotations

import pytest

from xdart.modules.ewald.arch_series import _IndexedList


class TestReadOpsMatchList:
    """Reading from an _IndexedList behaves identically to a list."""

    def test_iteration(self):
        il = _IndexedList([3, 1, 4, 1, 5])
        assert list(il) == [3, 1, 4, 1, 5]

    def test_indexing(self):
        il = _IndexedList([3, 1, 4])
        assert il[0] == 3
        assert il[-1] == 4

    def test_slicing_returns_plain_list(self):
        """list slicing is documented as returning a plain list, not
        the subclass — so callers that need the indexed behavior
        wrap explicitly."""
        il = _IndexedList([1, 2, 3, 4])
        sub = il[1:3]
        assert sub == [2, 3]
        # Slice is a plain list — _IndexedList doesn't override
        # __getitem__ for slices.
        assert type(sub) is list

    def test_len(self):
        il = _IndexedList([1, 2, 3])
        assert len(il) == 3

    def test_equality_with_list(self):
        il = _IndexedList([1, 2, 3])
        assert il == [1, 2, 3]
        assert il != [1, 2, 3, 4]


class TestMembershipIsFastAndCorrect:
    def test_in_returns_true_for_present(self):
        il = _IndexedList([10, 20, 30])
        assert 10 in il
        assert 20 in il
        assert 30 in il

    def test_in_returns_false_for_absent(self):
        il = _IndexedList([10, 20])
        assert 5 not in il
        assert 25 not in il

    def test_membership_after_mutations(self):
        il = _IndexedList()
        il.append(1)
        il.append(2)
        assert 1 in il and 2 in il and 3 not in il
        il.remove(1)
        assert 1 not in il and 2 in il


class TestMutationsKeepSetInSync:
    def test_append(self):
        il = _IndexedList()
        il.append(7)
        assert 7 in il
        assert list(il) == [7]

    def test_extend(self):
        il = _IndexedList([1])
        il.extend([2, 3, 4])
        for v in (1, 2, 3, 4):
            assert v in il

    def test_insert(self):
        il = _IndexedList([1, 3])
        il.insert(1, 2)
        assert list(il) == [1, 2, 3]
        assert 2 in il

    def test_remove(self):
        il = _IndexedList([1, 2, 3])
        il.remove(2)
        assert list(il) == [1, 3]
        assert 2 not in il

    def test_remove_with_duplicates_keeps_set_entry(self):
        """If duplicates exist, removing one shouldn't drop the set
        entry — the element is still present (the other copy)."""
        il = _IndexedList([1, 2, 2, 3])
        il.remove(2)
        assert list(il) == [1, 2, 3]
        # set must still report 2 as present.
        assert 2 in il

    def test_pop(self):
        il = _IndexedList([1, 2, 3])
        x = il.pop()
        assert x == 3
        assert 3 not in il

    def test_pop_with_index(self):
        il = _IndexedList([1, 2, 3])
        x = il.pop(0)
        assert x == 1
        assert 1 not in il
        assert list(il) == [2, 3]

    def test_clear(self):
        il = _IndexedList([1, 2, 3])
        il.clear()
        assert list(il) == []
        assert 1 not in il and 2 not in il and 3 not in il

    def test_setitem_replaces(self):
        il = _IndexedList([1, 2, 3])
        il[1] = 99
        assert list(il) == [1, 99, 3]
        assert 2 not in il
        assert 99 in il

    def test_setitem_slice_rebuilds_set(self):
        il = _IndexedList([1, 2, 3, 4])
        il[1:3] = [20, 30, 40]
        assert list(il) == [1, 20, 30, 40, 4]
        assert 2 not in il and 3 not in il
        assert all(v in il for v in (1, 20, 30, 40, 4))

    def test_delitem(self):
        il = _IndexedList([1, 2, 3])
        del il[1]
        assert list(il) == [1, 3]
        assert 2 not in il

    def test_delitem_slice(self):
        il = _IndexedList([1, 2, 3, 4, 5])
        del il[1:3]
        assert list(il) == [1, 4, 5]
        assert 2 not in il and 3 not in il
        assert 4 in il and 5 in il

    def test_sort_in_place_keeps_set(self):
        il = _IndexedList([3, 1, 2])
        il.sort()
        assert list(il) == [1, 2, 3]
        # Set unaffected.
        for v in (1, 2, 3):
            assert v in il


class TestEmptyAndConstruction:
    def test_empty(self):
        il = _IndexedList()
        assert list(il) == []
        assert 0 not in il

    def test_from_iterable(self):
        il = _IndexedList(range(5))
        assert list(il) == [0, 1, 2, 3, 4]
        for v in range(5):
            assert v in il
        assert 5 not in il
