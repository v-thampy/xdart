"""Build the accepted ``FrozenRunConfiguration`` a routed worker call consumes.

O-1a-W1R-D2 (review §43.4): worker execution takes its run policy as an explicit
argument -- the exact object the wrapper admitted and the entry gate qualified --
so a harness states that policy HERE, in one accepted value object, instead of
writing a mutable mirror attribute onto its host.

``run_options`` carries the run-scoped switches (``xye_only``,
``series_average``, ``meta_ext``); everything else is a named ``RunIntent``
field.
"""

from __future__ import annotations

from typing import Any


_ADMISSION_FIELDS = frozenset({
    "run_configuration",
    "_admitted_run_configuration",
    "run_configuration_floor",
})


def accepted_run(**fields: Any):
    """One frozen run configuration, from the production freeze owner."""
    from xrd_tools.session import RunIntent

    return RunIntent(**fields).freeze()


def gi_intent(**fields: Any):
    """A ``GIIntent`` for :func:`accepted_run`'s ``gi=`` field."""
    from xrd_tools.session import GIIntent

    return GIIntent(**fields)


def series_source(member, *, root=None):
    """A frozen-shaped numbered-series selection naming *member*.

    Mirrors what ``image_series_spec`` freezes (the containing directory as the
    uri, the picked member in options) without touching the filesystem, so a
    harness can name a member that need not exist.
    """
    from pathlib import Path

    from xrd_tools.core.scan import SourceKind, SourceSpec

    member = str(member)
    return SourceSpec(
        str(root) if root is not None else str(Path(member).parent),
        SourceKind.TIFF_SERIES,
        options={"selected_file": member},
    )


def directory_source(root, *, ext="tif", recursive=False, name_filter=None):
    """A frozen-shaped Image Directory selection."""
    from xrd_tools.sources import DirectorySourceSpec

    suffixes = ext if isinstance(ext, (tuple, list)) else (f".{str(ext).lstrip('.')}",)
    return DirectorySourceSpec(
        root=root,
        recursive=recursive,
        suffixes=tuple(suffixes),
        name_filter=name_filter,
    )


def single_image_source(path):
    """A frozen-shaped Single Image selection."""
    from xrd_tools.core.scan import SourceKind, SourceSpec

    return SourceSpec(str(path), SourceKind.IMAGE_FILE)


def container_source(path):
    """A frozen-shaped container Image Series selection (Eiger master/NeXus)."""
    from pathlib import Path

    from xrd_tools.core.scan import SourceKind, SourceSpec

    kind = (SourceKind.NEXUS_STACK
            if Path(str(path)).suffix.lower() == ".nxs"
            else SourceKind.EIGER_MASTER)
    return SourceSpec(str(path), kind)


def threshold_intent(**fields: Any):
    """A ``ThresholdIntent`` for :func:`accepted_run`'s ``threshold=`` field."""
    from xrd_tools.session import ThresholdIntent

    return ThresholdIntent(**fields)


def admitted_worker(target=None, /, *, frozen=None, **fields):
    """Publish an accepted configuration onto *target* through the REAL seam.

    O-1b-0.1.  Four independent test families created the impossible state
    "carrier published, admission ledger absent": the GI streaming rigs, the
    batch-finish run-end host, the live-refresh append-seed host, and the O-1b
    frame-boundary rig.  Every one of them was then refused by an
    identity-qualified consumer (review §45.3 for the host reads, §46.2 for the
    sink) or silently fell back to a default.

    Binding through ``wranglerWidget._bind_admitted_run_configuration`` -- the
    same owner production admission uses -- makes carrier, admission ledger and
    generation floor unable to drift, because a fixture can no longer set one
    without the others.

    ``target`` may be an existing worker/stub (bound in place and returned) or
    omitted, in which case a fresh ``SimpleNamespace`` is created.  Pass a
    prebuilt object as ``frozen=``, or the :func:`accepted_run` fields inline.

    Tests that deliberately construct a MALFORMED state -- a bare carrier with no
    ledger, a foreign identity, a stale generation -- must keep doing so by hand;
    this helper exists for the positive admitted-run path only.
    """
    from types import SimpleNamespace

    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )

    reserved = sorted(_ADMISSION_FIELDS.intersection(fields))
    if reserved:
        raise TypeError(
            "admitted_worker owns admission fields: "
            + ", ".join(reserved)
        )
    if frozen is None:
        frozen = accepted_run(**fields)
    elif fields:
        raise TypeError("pass either frozen= or accepted_run fields, not both")
    if target is None:
        target = SimpleNamespace()
    wranglerWidget._bind_admitted_run_configuration(target, frozen)
    return target
