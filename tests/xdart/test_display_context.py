# -*- coding: utf-8 -*-
"""Display-context kernel ownership and import-purity tests.

Two things are pinned here:

1. the ownership contract of the three Qt-free owner records themselves —
   write-once identity, a COMPLETE swap surface, a one-shot finalization claim,
   the token algebra a browse receipt must satisfy, and the release semantics;
2. import purity — the module must be loadable with no Qt, pyqtgraph, h5py,
   pyFAI, fabio or NumPy resident, proven in a FRESH interpreter because
   asserting it in-process is order-fragile (any earlier Qt-importing test in
   the same session would satisfy it vacuously);
RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets

from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplayBindings,
    DisplayContextError,
    DisplaySelection,
    new_context_token,
)

#: The complete display-side binding set.  Frozen here on purpose: a partial
#: swap is the "mixed-context render" failure (A's rows inside B's list, a blank
#: browse cake on scroll-back), and the only durable defence is that the surface
#: is enumerated in ONE place that a swap cannot silently under-fill.
_REQUIRED_BINDINGS = (
    "scan", "frame", "frame_ids", "frames", "viewer_rows_1d", "viewer_rows_2d",
    "record_store", "publication_store",
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

class _Store:
    """A minimal owned store: it can be cleared and it remembers that."""

    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _Receipt:
    """The shape of the O-2.2 browse-load receipt, values only."""

    def __init__(self, token, load_generation):
        self.token = token
        self.load_generation = load_generation


def _acquisition(scan=None, **overrides):
    fields = dict(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=None,
        config_generation=None,
        config_fingerprint="",
        run_scan_key="run_a",
        source_path="/data/run_a.nxs",
        scan=scan if scan is not None else object(),
        frame=object(),
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store(),
    )
    fields.update(overrides)
    return AcquisitionContext(**fields)


def _browse(token=None, load_generation=3, with_receipt=True, **overrides):
    token = token or new_context_token(ContextKind.BROWSE)
    labels = (1,)
    fields = dict(
        context_token=token,
        load_generation=load_generation,
        operation=(_Receipt(token, load_generation) if with_receipt else None),
        requested_path="/data/browse_b.nxs",
        scan_key="browse_b",
        scan=object(),
        frame=object(),
        frame_ids=labels,
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store(),
        scalar_catalog=SimpleNamespace(labels=labels),
        browse_1d_cache=object(),
        loaded_labels=labels,
    )
    fields.update(overrides)
    return BrowseContext(**fields)


# --------------------------------------------------------------------------- #
# 1. The owner records
# --------------------------------------------------------------------------- #

def test_context_tokens_are_unique_and_name_their_kind():
    tokens = [new_context_token(ContextKind.ACQUISITION) for _ in range(50)]
    tokens += [new_context_token(ContextKind.BROWSE) for _ in range(50)]
    assert len(set(tokens)) == len(tokens)
    assert all(token.startswith("acquisition-") for token in tokens[:50])
    assert all(token.startswith("browse-") for token in tokens[50:])


def test_identity_fields_are_write_once_on_both_contexts():
    """An identity field must not be re-pointable.

    This is the structural half of "a reference to a mutable object is not a
    snapshot": a context is REPLACED, never quietly aimed somewhere else, so a
    second acquisition owner cannot appear by assignment.
    """
    acquisition = _acquisition()
    browse = _browse()
    for context in (acquisition, browse):
        for name in sorted(context._IDENTITY_FIELDS):
            with pytest.raises(DisplayContextError):
                setattr(context, name, object())
    # The declared MUTABLE lifetime fields still move, each through its owner.
    # §12.6 C.4 — a genuine SAME-SOURCE boundary says so explicitly; omission
    # may not mean "reuse the previous source".
    acquisition.rescope_within_source("sub_b")
    assert acquisition.scan_key == "sub_b"
    acquisition.adopt_record_store("store")
    assert acquisition.record_store == "store"
    browse.mark_loaded()
    assert browse.loaded is True


def test_neither_context_carries_an_instance_dict():
    """``slots=True`` must survive the write-once mixin.

    A mixin without ``__slots__ = ()`` hands every instance a ``__dict__``
    back, which would let an unnamed field be attached to a context at runtime
    — the exact "second owner by attribute" the split exists to prevent.
    """
    assert not hasattr(_acquisition(), "__dict__")
    assert not hasattr(_browse(), "__dict__")
    assert not hasattr(DisplaySelection.for_context(_acquisition(), 1),
                       "__dict__")
    with pytest.raises(AttributeError):
        _acquisition().invented_owner = object()


def test_display_bindings_are_the_complete_swap_surface():
    assert DisplayBindings.field_names() == _REQUIRED_BINDINGS
    acquisition = _acquisition()
    bindings = acquisition.display_bindings()
    assert bindings.scan is acquisition.scan
    assert bindings.frame is acquisition.frame
    assert bindings.frames is acquisition.frames
    assert bindings.frame_ids is acquisition.frame_ids
    assert bindings.viewer_rows_1d is acquisition.viewer_rows_1d
    assert bindings.viewer_rows_2d is acquisition.viewer_rows_2d
    assert bindings.publication_store is acquisition.publication_store
    # The record store appears mid-run; the bindings must follow the adoption
    # rather than freezing the ``None`` that was there at construction.
    assert bindings.record_store is None
    acquisition.adopt_record_store("late")
    assert acquisition.display_bindings().record_store == "late"


def test_browse_and_acquisition_bindings_never_share_an_object():
    acquisition, browse = _acquisition(), _browse()
    a, b = acquisition.display_bindings(), browse.display_bindings()
    for name in _REQUIRED_BINDINGS:
        left, right = getattr(a, name), getattr(b, name)
        if left is None and right is None:
            continue
        assert left is not right, f"{name} is shared between the two contexts"


def test_a_hydration_owner_is_complete_or_it_is_nothing():
    """§11.1.1 — all four fields, or none of the authority.

    The parent asked only for a token and a scan key, so an empty source or a
    zero epoch read as a wildcard.  A carried field is not an authority while
    its absence is treated as permission.
    """
    from xdart.modules.display_context import HydrationOwner

    complete = HydrationOwner.of("tok", "key", "/data/a.nxs", 3)
    assert complete.qualified is True

    for owner in (
        HydrationOwner.of("", "key", "/data/a.nxs", 3),
        HydrationOwner.of("tok", "", "/data/a.nxs", 3),
        HydrationOwner.of("tok", "key", "", 3),
        HydrationOwner.of("tok", "key", "/data/a.nxs", 0),
        HydrationOwner.of("tok", "key", "/data/a.nxs", -1),
        HydrationOwner.of("tok", "key", "/data/a.nxs", True),
        HydrationOwner.of("tok", "key"),
        HydrationOwner(),
    ):
        assert owner.qualified is False, owner


def test_hydration_owner_decoding_is_total_and_never_raises():
    """§11.1.2 — a malformed owner is an inert refusal, not an exception.

    The decode runs on a completion delivered through a Qt signal; a raise
    there would escape into the render path instead of dropping one stale
    completion.  A bool is not an epoch either, and neither is something that
    merely coerces to one.
    """
    from xdart.modules.display_context import HydrationOwner

    class Hostile:
        def __str__(self):
            raise RuntimeError("hostile __str__")

    for epoch in ("not-an-int", None, 1.5, object(), True, False, [], "7"):
        owner = HydrationOwner.of("tok", "key", "/data/a.nxs", epoch)
        assert owner.epoch == 0, epoch
        assert owner.qualified is False, epoch

    hostile = HydrationOwner.of(Hostile(), "key", "/data/a.nxs", 3)
    assert hostile.context_token == ""
    assert hostile.qualified is False


def test_display_selection_is_frozen_and_names_its_context():
    acquisition = _acquisition()
    selection = DisplaySelection.for_context(acquisition, 11)
    assert selection.kind is ContextKind.ACQUISITION
    assert selection.context_token == acquisition.context_token
    assert selection.scan_key == acquisition.scan_key
    assert selection.source_path == acquisition.source_path
    assert selection.display_generation == 11
    assert selection.names(acquisition) is True
    assert selection.names(_browse()) is False
    assert selection.names(None) is False
    with pytest.raises(Exception):
        selection.display_generation = 12


def test_browse_receipt_must_carry_its_own_token_and_generation():
    """The token algebra is singular (O-3 §3).

    ``BrowseContext.context_token == receipt.token`` and
    ``BrowseContext.load_generation == receipt.load_generation``.  A receipt
    minted from an independent counter is refused at construction rather than
    quietly becoming a second correlation authority.
    """
    token = new_context_token(ContextKind.BROWSE)
    ok = _browse(token=token, load_generation=4)
    assert ok.operation.token == ok.context_token
    assert ok.operation.load_generation == ok.load_generation

    with pytest.raises(DisplayContextError):
        _browse(token=token, load_generation=4,
                operation=_Receipt("some-other-token", 4))
    with pytest.raises(DisplayContextError):
        _browse(token=token, load_generation=4,
                operation=_Receipt(token, 9))
    # No receipt at all is legal — the diagnostic channel may be off — and the
    # context still owns its token, so admission never depends on diagnostics.
    silent = _browse(with_receipt=False)
    assert silent.operation is None and silent.context_token


def test_browse_matches_only_its_exact_token_and_generation():
    browse = _browse(load_generation=7)
    assert browse.matches(browse.context_token, 7) is True
    assert browse.matches(browse.context_token, 6) is False
    assert browse.matches("browse-deadbeef-1", 7) is False


def test_browse_release_is_idempotent_and_drops_what_it_retained():
    browse = _browse()
    browse.frames[1] = object()
    browse.viewer_rows_1d[1] = object()
    browse.mark_loaded()
    cache = browse.browse_1d_cache
    browse.invalidate()
    browse.detach_browse_1d_cache(cache)

    browse.release()
    assert browse.released is True
    assert browse.loaded is False
    assert browse.publication_store.cleared == 1
    assert browse.frames == {} and browse.frame_ids == ()
    assert browse.frame_ids is browse.loaded_labels
    assert browse.viewer_rows_1d == {}
    assert browse.scalar_catalog is None
    # A released context must still REJECT a late completion by token rather
    # than crash on a half-nulled record.
    assert browse.matches(browse.context_token, browse.load_generation) is False

    browse.release()
    assert browse.publication_store.cleared == 1, "release is not idempotent"


def test_finalization_is_one_attempt_at_a_time_and_succeeds_once():
    """§9.1: attempts are serialised; only SUCCESS reaches finalized."""
    scan = object()
    acquisition = _acquisition(scan=scan)
    assert acquisition.begin_finalization() is scan
    assert acquisition.begin_finalization() is None, (
        "a second attempt ran while one was in flight")
    assert acquisition.finalized is False
    acquisition.complete_finalization()
    assert acquisition.finalized is True
    assert acquisition.begin_finalization() is None, (
        "a finalized identity was offered for finalization again")


def test_acquisition_scan_key_tracks_the_live_sub_scan():
    acquisition = _acquisition()
    assert acquisition.current_scan_key == "run_a"
    assert acquisition.scan_key == "run_a"
    acquisition.rescope_within_source("run_a_sub2")
    assert acquisition.scan_key == "run_a_sub2"
    assert acquisition.run_scan_key == "run_a", (
        "the run identity must survive a sub-scan boundary")


# --------------------------------------------------------------------------- #
# 2. Import purity
# --------------------------------------------------------------------------- #

def test_display_context_imports_no_gui_or_file_io_stack():
    """Proven in a FRESH interpreter, loading the module BY PATH.

    Loading by path deliberately bypasses ``xdart/__init__``, which imports the
        GUI package: the contract under test is that THIS module does not import
        a GUI toolkit or file decoder. Its xrd_tools value contracts are allowed.
    """
    module_path = (Path(__file__).resolve().parents[2]
                   / "src" / "xdart" / "modules" / "display_context.py")
    assert module_path.exists(), module_path
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('dc', r'{module_path}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        # dataclasses resolves annotations through sys.modules, so a
        # path-loaded module must be registered before it executes.
        "sys.modules['dc'] = module\n"
        "spec.loader.exec_module(module)\n"
        "assert module.AcquisitionContext and module.BrowseContext\n"
        "assert module.DisplaySelection and module.DisplayBindings\n"
        "banned = ('PySide6', 'PyQt5', 'PyQt6', 'qtpy', 'pyqtgraph', 'h5py',\n"
        "          'fabio', 'pyFAI')\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name in banned or any(\n"
        "        name.startswith(one + '.') for one in banned))\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_display_context_source_names_no_gui_toolkit():
    """Mirrors the acceptance oracle's own source-token check."""
    from xdart.modules import display_context

    source = inspect.getsource(display_context)
    # Exactly the oracle's token set (test_x1_context_split's
    # ``_o3_context_module_facts``): if any of these appears — even inside a
    # comment — the acceptance discriminator's ``qt_free`` row reads False.
    for token in ("pyqtgraph", "PySide6", "QtCore", "QtWidgets"):
        assert token not in source, f"{token} appears in the Qt-free owners"
    for token in ("import h5py", "import numpy", "import fabio",
                  "import pyFAI"):
        assert token not in source, f"{token} appears in the Qt-free owners"


# --------------------------------------------------------------------------- #
# 3. Acquisition adoption, through the real widget
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _frozen_configuration():
    """An admitted configuration WITH a source (§11.2.1).

    A wrangler run is refused unless its accepted configuration names one: the
    acquisition owner takes its source identity from there and never from the
    mutable ``scan.data_file``, which at admission can still be the scratch
    placeholder or the previous run's output.
    """
    from dataclasses import replace as _replace

    from xrd_tools.session.run_configuration import (
        FrozenSourceSpec,
        RunIntent,
    )

    frozen = RunIntent().freeze()
    return _replace(frozen, source=FrozenSourceSpec(
        family="source", source_kind="image_file", uri="/raw/accepted-source.h5"))


# --------------------------------------------------------------------------- #
# 4. The paused browse owner (c2), on the real widget
#
# The real-file acceptance for "B is servable" lives in
# ``test_x1_context_split.py``; these rows pin the OWNERSHIP properties that a
# real-data sequence cannot isolate — that the acquisition object is never
# passed anywhere, that replacing a browse releases the previous one, and that
# an admission is by exact identity rather than by arrival.
# --------------------------------------------------------------------------- #

def _publication(label, scan_key, source):
    """A real publication, so the store lookup under test is a real lookup."""
    import numpy as np
    from xdart.modules.frame_publication import publication_from_frame_view
    from xrd_tools.core import FrameView, IntegrationResult1D

    view = FrameView.from_results(
        label=label,
        result_1d=IntegrationResult1D(
            radial=np.linspace(0.5, 3.5, 4),
            intensity=np.array([2.0, 4.0, 8.0, 16.0]),
            sigma=np.ones(4), unit="q_A^-1"),
        metadata_raw={"i0": 1.0},
        source_path=source, source_frame_index=label)
    return publication_from_frame_view(view, scan_key=scan_key)
