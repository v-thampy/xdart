"""H17 — source-registry seam: the ``register_source`` extension point + the
``guess_source_kind`` dispatch table.

Pins the seam that made Bluesky/NXWriter additive (and Tiled next): a new format
is a single ``register_source(kind, factory)`` plus one ``guess_source_kind``
arm, never a rewrite of ``open_source``.  This is the headless seam the planned
plug-and-play source-format registry (post-v1.1) builds on, so a regression here
is a regression in "adding a detector/format is a registration, not a hunt."

Extension policy (worked example: Bluesky):
  1. add a ``SourceKind`` member (already: ``NEXUS_STACK`` covers Bluesky, ``TILED`` reserved);
  2. teach ``guess_source_kind`` to map the URI/extension/content to that kind;
  3. either add a built-in arm in ``open_source`` OR (preferred for out-of-tree
     formats) ``register_source(kind, factory)`` — the registry is consulted
     BEFORE the built-in dispatch, so a registered factory OVERRIDES it.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources import guess_source_kind, open_source, register_source
from xrd_tools.sources.registry import _REGISTRY


@contextlib.contextmanager
def _isolated_registry():
    """Save/restore the process-global source registry so a test's
    ``register_source`` never leaks into the rest of the suite."""
    saved = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


class _FakeSource:
    """Minimal FrameSource duck for the registry seam (no I/O)."""

    def __init__(self, spec):
        self.spec = spec
        self.frame_indices = range(1)

    def load_frame(self, index):
        return np.zeros((2, 2))


# ---- the register_source extension seam ------------------------------------

def test_register_source_factory_is_used_for_its_kind():
    """A registered factory opens sources of its kind — the out-of-tree hook."""
    with _isolated_registry():
        register_source(SourceKind.TILED, lambda spec: _FakeSource(spec))
        src = open_source(SourceSpec("tiled://run/1", SourceKind.TILED))
        assert isinstance(src, _FakeSource)


def test_register_source_overrides_the_builtin_dispatch():
    """The registry is consulted BEFORE the built-in if-chain, so a registered
    factory OVERRIDES the built-in opener for that kind (lets a site swap an
    implementation without editing open_source)."""
    with _isolated_registry():
        register_source(SourceKind.IMAGE_FILE, lambda spec: _FakeSource(spec))
        src = open_source(SourceSpec("/x.tif", SourceKind.IMAGE_FILE))
        assert isinstance(src, _FakeSource)   # not the built-in ImageFileSource


def test_register_source_is_additive():
    """Registering one kind's factory does not disturb another kind's dispatch."""
    with _isolated_registry():
        register_source(SourceKind.TILED, lambda spec: _FakeSource(spec))
        # A kind with no registered factory still raises the built-in clean error
        # (i.e. the registration did not swallow the rest of the dispatch).
        with pytest.raises(ValueError):
            open_source(SourceSpec("/x.weird", SourceKind.UNKNOWN))


def test_register_source_accepts_string_kind():
    """``register_source`` coerces a string kind (plugin authors may pass one)."""
    with _isolated_registry():
        register_source("tiled", lambda spec: _FakeSource(spec))
        assert isinstance(
            open_source(SourceSpec("tiled://x", SourceKind.TILED)), _FakeSource)


# ---- open_source contract --------------------------------------------------

def test_open_source_passes_through_an_existing_framesource():
    fs = _FakeSource(SourceSpec("mem://x", SourceKind.MEMORY))
    assert open_source(fs) is fs


def test_open_source_unknown_kind_raises_cleanly():
    with pytest.raises(ValueError):
        open_source(SourceSpec("/x.weird", SourceKind.UNKNOWN))


# ---- guess_source_kind dispatch table --------------------------------------

def test_guess_source_kind_directory_is_a_tiff_series(tmp_path):
    d = tmp_path / "series"
    d.mkdir()
    assert guess_source_kind(d) is SourceKind.TIFF_SERIES


def test_guess_source_kind_hdf5_family_is_a_nexus_stack(tmp_path):
    # Extension fallback: even a non-HDF5 file with an h5/nexus extension routes
    # to NEXUS_STACK (the opener validates content); pins the container policy.
    for ext in (".nxs", ".h5", ".hdf5", ".cxi"):
        p = tmp_path / f"f{ext}"
        p.write_text("placeholder")
        assert guess_source_kind(p) is SourceKind.NEXUS_STACK, ext


def test_guess_source_kind_tif_is_an_image_file(tmp_path):
    p = tmp_path / "f.tif"
    p.write_bytes(b"II*\x00")  # minimal TIFF-ish header; classification is by ext
    assert guess_source_kind(p) is SourceKind.IMAGE_FILE


# ---- the Bluesky worked example (real file; skip without test data) --------

_REAL = __import__("pathlib").Path(
    __import__("os").environ.get(
        "XDART_TEST_DATA",
        __import__("pathlib").Path(__file__).resolve().parents[2] / "test_data")
) / "nexus" / "LaB6_0710_1025pm_00005.nxs"


@pytest.mark.skipif(not _REAL.exists(), reason=f"real Bluesky file not found: {_REAL}")
def test_guess_source_kind_bluesky_nxs_is_a_nexus_stack():
    """A real Bluesky/NXWriter .nxs classifies as NEXUS_STACK and opens through
    the seam — the additive-format case H17 exists to keep working."""
    assert guess_source_kind(_REAL) is SourceKind.NEXUS_STACK
    src = open_source(SourceSpec(_REAL, SourceKind.NEXUS_STACK, entry="entry"))
    assert len(list(src.frame_indices)) == 3
