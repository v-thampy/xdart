# -*- coding: utf-8 -*-
"""Boolean filename-filter expressions (F1).

One shared, headless ``compile_filter`` used by every GUI "Filter" field
(Image Directory glob, Eiger ``_master.h5`` queue, BG Match) so all sites
share one grammar and one set of unit tests.  Sites glob only the literal
suffix (``*.tif`` / ``*_master.h5``) and apply the compiled predicate to
the names in Python — the filter is no longer encoded into the glob.

Grammar (MVP — parentheses deliberately not operators, they occur in real
filenames):

* bare space-separated terms — **unordered AND** of case-insensitive
  substring matches: ``abc def`` matches names containing both, any order
  (the pre-1.0 glob was an *ordered* AND; single-term filters behave
  identically);
* ``|`` (or the uppercase word ``OR``) — union of AND-clauses:
  ``abc | def`` matches names containing either;
* leading ``-term`` (or the uppercase word ``NOT`` before a term) —
  exclusion: ``abc -bg`` matches names with ``abc`` and without ``bg``;
* ``OR``/``NOT`` are operators only in UPPERCASE — lowercase ``or``/``not``
  stay ordinary substrings;
* empty/whitespace-only expression matches everything.

Malformed expressions (dangling ``NOT``, empty ``|`` branch, bare ``-``)
raise :class:`ValueError` — GUI callers catch it and fall back/surface it.
This module must stay import-light (pure Python): it is part of the
Qt-free ``xrd_tools.core`` surface.
"""
from __future__ import annotations

from typing import Callable

__all__ = ["compile_filter"]


def _parse(expr: str | None) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Parse to OR-clauses of (positive terms, negative terms), lowered."""
    if expr is None:
        return []
    # '|' is an operator even without surrounding spaces ("a|b"); it cannot
    # legitimately appear in filenames on Windows and is vanishingly rare
    # elsewhere.
    tokens = str(expr).replace("|", " | ").split()
    if not tokens:
        return []

    clauses: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    pos: list[str] = []
    neg: list[str] = []
    branch_seen = False
    negate_next = False

    def _close_branch() -> None:
        nonlocal pos, neg, branch_seen
        if not branch_seen:
            raise ValueError(
                "empty OR branch — every '|' / 'OR' needs a term on both sides"
            )
        clauses.append((tuple(pos), tuple(neg)))
        pos, neg = [], []
        branch_seen = False

    for tok in tokens:
        if tok == "|" or tok == "OR":
            if negate_next:
                raise ValueError("NOT must be followed by a term, not an OR")
            _close_branch()
            continue
        if tok == "NOT":
            if negate_next:
                raise ValueError("doubled NOT")
            negate_next = True
            continue
        term = tok
        negated = negate_next
        negate_next = False
        if not negated and term.startswith("-"):
            if term == "-":
                raise ValueError("bare '-' — exclusion needs a term: -term")
            negated, term = True, term[1:]
        (neg if negated else pos).append(term.lower())
        branch_seen = True

    if negate_next:
        raise ValueError("NOT must be followed by a term")
    _close_branch()
    return clauses


def compile_filter(expr: str | None) -> Callable[[str], bool]:
    """Compile a filter expression into a name predicate.

    Returns a pure, thread-safe ``Callable[[str], bool]`` (case-insensitive).
    See the module docstring for the grammar.  Raises :class:`ValueError`
    on a malformed expression.
    """
    clauses = _parse(expr)
    if not clauses:
        return lambda name: True

    def predicate(name: str) -> bool:
        low = str(name).lower()
        return any(
            all(term in low for term in positives)
            and not any(term in low for term in negatives)
            for positives, negatives in clauses
        )

    return predicate
