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
