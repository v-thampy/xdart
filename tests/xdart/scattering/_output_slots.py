# -*- coding: utf-8 -*-
"""Where a run actually writes, given the save path it was REQUESTED with.

Under ADR-0010 the public name is ``<family><slot>.nexus``, and an explicit save
path supplies the FAMILY, so a run requested at ``processed.nexus`` in ``Int 2D``
writes ``processed_int2d.nexus``.  Tests hand the request to the intent and then
inspect the written file, so the two must be spelled apart.

ONE spelling, shared.  Three test modules needed this and the alternative was
three copies of the same table -- the identical defect class as the duplicated
family pattern in ``finite_artifact.py`` and the duplicated slot suffixes in the
GUI, both of which had already gone wrong once.  Derived from the production
table rather than restated, so a change to the slot vocabulary fails here
instead of silently disagreeing with it.
"""

from __future__ import annotations

from pathlib import Path


def written(target: Path | str, processing_mode: str = "Int 1D") -> Path:
    """The artifact a run REQUESTED at *target* publishes under *processing_mode*."""
    from xdart.gui.tabs.scattering.output_preflight import _RUN_MODE_SLOTS

    requested = Path(target)
    return requested.with_name(
        f"{requested.stem}{_RUN_MODE_SLOTS[processing_mode]}.nexus"
    )
