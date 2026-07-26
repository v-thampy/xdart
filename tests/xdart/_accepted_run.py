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
