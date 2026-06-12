"""F1 — boolean filename-filter expressions (xrd_tools.core.filters).

Headless unit tests for the ONE grammar every GUI Filter field shares
(Image Directory glob, Eiger _master.h5 queue, BG Match).
"""
import pytest

from xrd_tools.core.filters import compile_filter


# ── match-all defaults ───────────────────────────────────────────────────────

@pytest.mark.parametrize("expr", [None, "", "   "])
def test_empty_expression_matches_all(expr):
    pred = compile_filter(expr)
    assert pred("anything_at_all.tif")
    assert pred("")


# ── single term: identical to the old *term* glob ────────────────────────────

def test_single_term_substring():
    pred = compile_filter("abc")
    assert pred("xx_abc_yy")
    assert not pred("xx_ab_yy")


def test_single_term_case_insensitive():
    pred = compile_filter("ABC")
    assert pred("xx_abc_yy")
    assert compile_filter("abc")("XX_ABC_YY")


# ── AND: unordered (the deliberate change from the old ordered glob) ─────────

def test_multi_term_unordered_and():
    pred = compile_filter("abc def")
    assert pred("abc_then_def")
    assert pred("def_then_abc")          # old glob *abc*def* rejected this
    assert not pred("abc_only")


# ── OR ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr", ["abc | def", "abc OR def", "abc|def"])
def test_or_union(expr):
    pred = compile_filter(expr)
    assert pred("has_abc")
    assert pred("has_def")
    assert not pred("has_ghi")


def test_or_of_and_clauses():
    pred = compile_filter("a b | c d")
    assert pred("b_a")                   # first clause (unordered)
    assert pred("d_c")                   # second clause
    assert not pred("a_c")               # half of each


# ── NOT / exclusion ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr", ["-abc", "NOT abc"])
def test_exclusion(expr):
    pred = compile_filter(expr)
    assert not pred("has_abc")
    assert pred("has_def")


def test_and_with_exclusion():
    pred = compile_filter("abc -bg")
    assert pred("abc_sample")
    assert not pred("abc_bg")
    assert not pred("plain_bg")


def test_exclusion_is_per_or_branch():
    pred = compile_filter("abc -bg | def")
    assert pred("def_bg")                # exclusion binds the FIRST branch only
    assert not pred("abc_bg")


# ── operators are uppercase-only; punctuation is literal ─────────────────────

def test_lowercase_or_not_are_plain_terms():
    assert compile_filter("or")("xx_or_yy")
    assert not compile_filter("or")("xx_yy")
    assert compile_filter("not")("can_not_open")


def test_hyphen_inside_term_is_literal():
    pred = compile_filter("my-scan")
    assert pred("xx_my-scan_yy")
    assert not pred("my_scan")


def test_parentheses_are_literal_filename_characters():
    pred = compile_filter("scan(1)")
    assert pred("scan(1)_0001")
    assert not pred("scan(2)_0001")


# ── malformed expressions raise ──────────────────────────────────────────────

@pytest.mark.parametrize("expr", [
    "abc |",          # trailing OR
    "| abc",          # leading OR
    "abc | | def",    # empty branch
    "abc NOT",        # dangling NOT
    "NOT | abc",      # NOT before OR
    "NOT NOT abc",    # doubled NOT
    "-",              # bare exclusion
])
def test_malformed_raises_value_error(expr):
    with pytest.raises(ValueError):
        compile_filter(expr)


# ── realistic filenames ──────────────────────────────────────────────────────

def test_real_eiger_names():
    pred = compile_filter("scan001 Eiger")
    assert pred("Eiger_scan001")         # unordered AND
    assert not pred("Eiger_scan002")


def test_bg_match_composition():
    # the BG site prepends the scan name: f"{scan_name} {bg_filter}"
    pred = compile_filter("LaB6_cal bg")
    assert pred("LaB6_cal_bg_0001")
    assert not pred("LaB6_cal_0001")
