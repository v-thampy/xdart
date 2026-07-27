# -*- coding: utf-8 -*-
"""X1 O-3 c1 — the display-context kernel and the acquisition adoption.

Three things are pinned here, and nothing else:

1. the ownership contract of the three Qt-free owner records themselves —
   write-once identity, a COMPLETE swap surface, a one-shot finalization claim,
   the token algebra a browse receipt must satisfy, and the release semantics;
2. import purity — the module must be loadable with no Qt, pyqtgraph, h5py,
   pyFAI, fabio or NumPy resident, proven in a FRESH interpreter because
   asserting it in-process is order-fragile (any earlier Qt-importing test in
   the same session would satisfy it vacuously);
3. the adoption itself, through the REAL offscreen ``staticWidget``: the run
   admission owner builds exactly ONE acquisition context from the objects the
   widget already owns, the deleted ``_x1_run_scan_capture`` alias is gone and
   nothing reintroduces an equivalent mutable acquisition alias, and the
   run-end finalizer drives the context rather than the alias.

RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

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
    fields = dict(
        context_token=token,
        load_generation=load_generation,
        operation=(_Receipt(token, load_generation) if with_receipt else None),
        requested_path="/data/browse_b.nxs",
        scan_key="browse_b",
        scan=object(),
        frame=object(),
        frame_ids=["1"],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store(),
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
    acquisition.rescope_to("sub_b")
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

    browse.release()
    assert browse.released is True
    assert browse.loaded is False
    assert browse.publication_store.cleared == 1
    assert browse.frames == {} and browse.frame_ids == []
    assert browse.viewer_rows_1d == {}
    # A released context must still REJECT a late completion by token rather
    # than crash on a half-nulled record.
    assert browse.matches(browse.context_token, browse.load_generation) is False

    browse.release()
    assert browse.publication_store.cleared == 1, "release is not idempotent"


def test_finalization_claim_is_one_shot():
    scan = object()
    acquisition = _acquisition(scan=scan)
    assert acquisition.claim_finalization() is scan
    assert acquisition.claim_finalization() is None
    assert acquisition.finalization_claimed is True
    assert acquisition.finalized is False
    acquisition.mark_finalized()
    assert acquisition.finalized is True


def test_acquisition_scan_key_tracks_the_live_sub_scan():
    acquisition = _acquisition()
    assert acquisition.current_scan_key == "run_a"
    assert acquisition.scan_key == "run_a"
    acquisition.rescope_to("run_a_sub2")
    assert acquisition.scan_key == "run_a_sub2"
    assert acquisition.run_scan_key == "run_a", (
        "the run identity must survive a sub-scan boundary")


# --------------------------------------------------------------------------- #
# 2. Import purity
# --------------------------------------------------------------------------- #

def test_display_context_imports_no_gui_or_science_stack():
    """Proven in a FRESH interpreter, loading the module BY PATH.

    Loading by path deliberately bypasses ``xdart/__init__``, which imports the
    GUI package: the contract under test is that THIS module's own imports are
    stdlib-only, not that the package around it is.
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
        "          'fabio', 'pyFAI', 'numpy', 'pandas', 'xdart', 'xrd_tools')\n"
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


@pytest.fixture
def widget(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    try:
        yield value
    finally:
        try:
            value._exit_run_state(value._new_projection_receipt())
        except Exception:
            pass
        try:
            value._controls_v2_refresh_timer.cancel()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def test_enter_run_state_builds_one_acquisition_context(widget):
    """The run-admission owner adopts the objects the widget ALREADY owns.

    Every field is checked by identity: a context that allocated its own frame
    map or its own store would be a second owner, not an owner record for the
    one the run already has.
    """
    assert widget._acquisition_context is None
    widget._enter_run_state()
    context = widget._acquisition_context

    assert context is not None
    assert context.kind is ContextKind.ACQUISITION
    assert context.scan is widget.scan
    assert context.frame is widget.frame
    assert context.frame_ids is widget.frame_ids
    assert context.frames is widget.frames
    assert context.viewer_rows_1d is widget.viewer_rows_1d
    assert context.viewer_rows_2d is widget.viewer_rows_2d
    assert context.publication_store is widget.publication_store
    assert context.context_token.startswith("acquisition-")
    assert context.current_scan_key == context.run_scan_key

    # Re-entry is a no-op: the run has ONE context, not one per fired signal.
    widget._enter_run_state()
    assert widget._acquisition_context is context


def test_context_refuses_a_configuration_that_was_not_admitted(
        widget, monkeypatch):
    """O-3 rule 8 — only the EXACT admitted object, never an equal-valued one.

    ``FrozenRunConfiguration`` equality is by content, so a reconstructed copy
    on the public carrier is indistinguishable from the admitted one by value.
    The context takes it only when the carrier IS the admission ledger.
    """
    from dataclasses import replace as _replace

    from xrd_tools.session.run_configuration import RunIntent

    intent = RunIntent()
    admitted = intent.freeze()
    wrangler = widget.wrangler

    # (a) admitted: carrier IS the ledger.
    monkeypatch.setattr(wrangler, "run_configuration", admitted, raising=False)
    monkeypatch.setattr(wrangler, "_admitted_run_configuration", admitted,
                        raising=False)
    widget._enter_run_state()
    context = widget._acquisition_context
    assert context.run_configuration is admitted
    assert context.config_generation == admitted.identity[0]
    assert context.config_fingerprint == admitted.identity[1]

    # (b) foreign but content-equal: refused, and NOT silently substituted.
    foreign = _replace(admitted)
    assert foreign == admitted and foreign is not admitted
    widget._exit_run_state(widget._new_projection_receipt())
    monkeypatch.setattr(wrangler, "run_configuration", foreign, raising=False)
    widget._enter_run_state()
    refused = widget._acquisition_context
    assert refused.run_configuration is None, (
        "an equal-valued foreign configuration was accepted as admitted")
    assert refused.config_generation is None
    assert refused.config_fingerprint == ""


def test_run_end_finalizes_through_the_context_and_releases_it(
        widget, monkeypatch):
    """The run-end substep drives the context, exactly once."""
    finished = []
    monkeypatch.setattr(
        widget.displayframe, "finish_processing",
        lambda scan=None, key=None: finished.append((scan, key)))

    widget._enter_run_state()
    run_scan = widget._acquisition_context.scan
    widget._exit_run_state(widget._new_projection_receipt())

    assert widget._acquisition_context is None
    assert len(finished) == 1
    assert finished[0][0] is run_scan
    # A second exit must not finalize the identity again.
    widget._exit_run_state(widget._new_projection_receipt())
    assert len(finished) == 1


def test_the_acquisition_alias_is_gone_and_has_no_replacement(widget):
    """O-3 rule 6 and the c3 discriminator's structural half.

    The alias must be absent AND no equivalent mutable acquisition alias may
    have taken its place: the guard is a bounded census over the production
    module's own assignment targets, not a general taint analysis.
    """
    from xdart.gui.tabs.static_scan import static_scan_widget as ssw_module

    widget._enter_run_state()
    assert not hasattr(widget, "_x1_run_scan_capture")

    source = Path(inspect.getfile(ssw_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # An ATTRIBUTE census, not a substring scan: the commit's own prose names
    # the retired alias to say why it is gone, and a comment is not a reader.
    revived = sorted({
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "_x1_run_scan_capture"
    })
    assert revived == [], "the deleted acquisition alias was reintroduced"

    # Any OTHER `self.<name> = self.scan` in the widget is a new mutable
    # acquisition alias by definition; the context is the only owner allowed.
    aliases = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "scan"
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                aliases.append((node.lineno, target.attr))
    assert aliases == [], f"new mutable acquisition aliases: {aliases}"


# --------------------------------------------------------------------------- #
# 4. The paused browse owner (c2), on the real widget
#
# The real-file acceptance for "B is servable" lives in
# ``test_x1_context_split.py``; these rows pin the OWNERSHIP properties that a
# real-data sequence cannot isolate — that the acquisition object is never
# passed anywhere, that replacing a browse releases the previous one, and that
# an admission is by exact identity rather than by arrival.
# --------------------------------------------------------------------------- #

def _paused_run(widget, monkeypatch):
    """Put the real widget in the state a paused browse is legal from."""
    widget._enter_run_state()
    widget.h5viewer.paused_browse_active = True
    # The file worker is not started: the queue is inspected directly, so the
    # task under test is the exact object production enqueued.
    queued = []
    monkeypatch.setattr(widget.h5viewer, "_ensure_file_thread_running",
                        lambda: None)
    monkeypatch.setattr(widget.h5viewer.file_thread.queue, "put", queued.append)
    return queued


def test_paused_browse_never_hands_over_the_acquisition_scan(
        widget, monkeypatch, tmp_path):
    """M1's guard: the browse target is a NEW scan, and A's own load entry
    points are never called.

    Spies wrap the REAL acquisition ``LiveScan``'s methods rather than replacing
    the seam under test, so the assertion is about production behaviour and not
    about a double.
    """
    queued = _paused_run(widget, monkeypatch)
    acquisition = widget._acquisition_context.scan
    calls = []
    for name in ("set_datafile", "reset", "load_from_h5"):
        original = getattr(acquisition, name, None)
        if callable(original):
            monkeypatch.setattr(
                acquisition, name,
                lambda *a, _n=name, _o=original, **k: (
                    calls.append(_n), _o(*a, **k))[1],
                raising=False)

    path = tmp_path / "browsed_result.nxs"
    context = widget._begin_paused_browse(str(path))

    assert context is not None
    assert context.scan is not acquisition, (
        "the browse was handed the acquisition scan itself")
    assert context.publication_store is not widget.publication_store
    assert context.scan_key == "browsed_result"
    assert calls == [], f"the browse called {calls} on the acquisition scan"

    assert len(queued) == 1
    task = queued[0]
    from xdart.gui.tabs.static_scan.scan_threads import BrowseLoadTask, FileTask

    assert isinstance(task, BrowseLoadTask) and isinstance(task, FileTask)
    assert task.method == "load_browse_datafile"
    assert task.scan is context.scan
    assert task.fname == str(path)
    assert task.scan_name == context.scan_key
    assert task.context_token == context.context_token
    assert task.load_generation == context.load_generation
    # The accepted O-2.2R envelope is EXTENDED, never widened.
    assert set(FileTask.__dataclass_fields__) == {"method", "operation"}


def test_replacing_a_browse_releases_the_previous_one(
        widget, monkeypatch, tmp_path):
    """One browse at a time; the loser's store is released and it can never be
    admitted afterwards."""
    queued = _paused_run(widget, monkeypatch)

    first = widget._begin_paused_browse(str(tmp_path / "first.nxs"))
    first_store = first.publication_store
    second = widget._begin_paused_browse(str(tmp_path / "second.nxs"))

    assert widget._browse_context is second
    assert first.released is True
    assert first_store.snapshot() == {}
    assert second.released is False
    assert first.context_token != second.context_token
    assert first.load_generation != second.load_generation
    assert len(queued) == 2

    # The FIRST task completing late must not be admitted, and must not touch
    # the newer selection.
    assert widget._on_browse_loaded(queued[0]) is None
    assert widget._display_selection is None
    assert widget.displayframe.scan is widget.scan


def test_admission_refuses_a_completion_it_does_not_own(
        widget, monkeypatch, tmp_path):
    """Exact identity, not arrival order: token, generation and a live run."""
    from xdart.gui.tabs.static_scan.scan_threads import BrowseLoadTask

    queued = _paused_run(widget, monkeypatch)
    context = widget._begin_paused_browse(str(tmp_path / "browsed.nxs"))
    task = queued[0]

    foreign_token = BrowseLoadTask(
        method="load_browse_datafile", operation=None, scan=context.scan,
        fname=task.fname, scan_name=task.scan_name,
        context_token="browse-not-ours", load_generation=task.load_generation)
    assert widget._on_browse_loaded(foreign_token) is None

    stale_generation = BrowseLoadTask(
        method="load_browse_datafile", operation=None, scan=context.scan,
        fname=task.fname, scan_name=task.scan_name,
        context_token=task.context_token,
        load_generation=task.load_generation + 1)
    assert widget._on_browse_loaded(stale_generation) is None
    assert widget._display_selection is None

    # The exact task IS admitted while the run is paused...
    assert widget._on_browse_loaded(task) is not None
    assert widget._display_selection.kind is ContextKind.BROWSE
    # ...and the same task is refused once the run is no longer active.
    widget._run_active = False
    widget._display_selection = None
    assert widget._on_browse_loaded(task) is None


def test_the_swap_moves_every_display_binding(widget, monkeypatch, tmp_path):
    """M6's guard: a partial swap is a mixed-context render.

    Every field of the frozen binding record must land on BOTH display-side
    consumers, and the never-swapped owners must not move.
    """
    queued = _paused_run(widget, monkeypatch)
    context = widget._begin_paused_browse(str(tmp_path / "browsed.nxs"))
    acquisition = widget._acquisition_context

    integrator_scan = widget.integratorTree.scan
    stitch_scan = widget.stitch_thread.scan
    file_thread_scan = widget.h5viewer.file_thread.scan
    widget._on_browse_loaded(queued[0])

    bindings = context.display_bindings()
    for target in (widget.h5viewer, widget.displayframe):
        for name in _REQUIRED_BINDINGS:
            if name == "record_store":
                continue
            assert getattr(target, name) is getattr(bindings, name), (
                f"{type(target).__name__}.{name} was not swapped")
    assert widget.displayframe.frame_record_store() is None, (
        "the acquisition record store is still reachable from the browse")

    # Never swapped: the widget's own scan, the workers, the file thread.
    assert widget.scan is acquisition.scan
    assert widget.publication_store is acquisition.publication_store
    assert widget.integratorTree.scan is integrator_scan
    assert widget.stitch_thread.scan is stitch_scan
    assert widget.h5viewer.file_thread.scan is file_thread_scan

    # And the selection names the context at the stamped generation.
    selection = widget._display_selection
    assert selection.names(context)
    assert selection.display_generation == \
        widget.displayframe.display_generation
    assert widget.displayframe.display_context_token == context.context_token


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


def test_store_first_reads_never_cross_the_selected_context(
        widget, monkeypatch, tmp_path):
    """M2's guard: the data tier must resolve BOTH stores from the selection.

    A and B publish the SAME label.  A label-only read — one that resolves the
    store from the widget instead of from the selection — serves A's frame for
    B's label, which is the pin/data asymmetry this seam exists to close.
    """
    queued = _paused_run(widget, monkeypatch)
    label = 1
    widget.publication_store.upsert(
        _publication(label, "run_a", "/data/run_a.nxs"))

    context = widget._begin_paused_browse(str(tmp_path / "browsed.nxs"))
    context.publication_store.upsert(
        _publication(label, context.scan_key, str(tmp_path / "browsed.nxs")))
    widget._on_browse_loaded(queued[0])

    record_store, publications = widget._display_selected_stores()
    assert record_store is None
    assert publications is context.publication_store
    assert publications is not widget.publication_store

    view = widget.store_first_frame_view(label)
    assert view is not None
    assert str(view.source_path).endswith("browsed.nxs"), (
        "the store-first read served the acquisition's frame for the "
        "browsed scan's label")
    # A's own publication is untouched and still its own.
    assert widget.publication_store.get(label).scan_key == "run_a"


def test_a_browse_request_without_a_canonical_name_is_refused(
        widget, monkeypatch):
    """Fail closed: no canonical scan name, no context and no queued task."""
    queued = _paused_run(widget, monkeypatch)
    assert widget._begin_paused_browse("") is None
    assert widget._browse_context is None
    assert queued == []


def test_idle_browse_takes_the_legacy_path(widget, monkeypatch, tmp_path):
    """With no run active the browse owner defers to the legacy seam.

    Idle browsing is deliberately out of scope: it keeps loading into the
    widget's own scan, and no ``BrowseContext`` is minted for it.
    """
    seen = []
    monkeypatch.setattr(widget.h5viewer, "set_file",
                        lambda path, **kw: seen.append(path))
    assert widget._run_active is False
    widget._begin_paused_browse(str(tmp_path / "idle.nxs"))
    assert seen == [str(tmp_path / "idle.nxs")]
    assert widget._browse_context is None


def test_only_the_context_owner_constructs_an_acquisition_context():
    """ONE construction site, so a second owner cannot appear by copy-paste."""
    from xdart.gui.tabs.static_scan import static_scan_widget as ssw_module

    source = Path(inspect.getfile(ssw_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    sites = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "AcquisitionContext"
    ]
    assert len(sites) == 1, f"AcquisitionContext is constructed at {sites}"

    owner = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_acquisition_context")
    assert owner.lineno < sites[0] < owner.end_lineno, (
        "the only construction site is outside the run-admission owner")
