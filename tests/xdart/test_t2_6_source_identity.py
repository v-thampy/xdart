"""T-2.6 (Correction C, partial): canonical source identity + request-local GI
hydration + typed source receipt (§15.12-C items 1/2/3/5).

Item 4 (proof-init false → unproved-empty ⇒ UNKNOWN) is delivered separately in
T-2.6b once Codex's amended §12 tripwire lands; only its targeted-empty ⇒
KNOWN_EMPTY half is asserted here (a regression guard, not a fail-before).

Each behavioural test below is red at the pre-fix tip (84c59d48) and green here.
Production-wired: real ``staticWidget`` / wrangler / ``DirectoryIndexSession``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _set_signal(widget, **fields):
    signal = widget.wrangler.parameters.child("Signal")
    for name, value in fields.items():
        signal.child(name).setValue(value)


# --------------------------------------------------------------------------
# Item 1/2 — canonical SourceSelectionIdentity (§15.11-C tests 1-4)
# --------------------------------------------------------------------------

def test_tiff_directory_ab_identity_differs(widget, tmp_path):
    """A non-container (TIFF) Image Directory A→B must yield DIFFERENT, non-None
    identities — the §15.5 gap where both collapsed to ``None`` and B inherited
    A's proved motor."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _set_signal(widget, inp_type="Image Directory", img_ext="tif", img_dir=str(a))
    token_a = widget._controls_v2_source_token()
    _set_signal(widget, img_dir=str(b))
    token_b = widget._controls_v2_source_token()

    assert token_a is not None
    assert token_b is not None
    assert token_a != token_b


def test_two_tiff_series_have_distinct_identities(widget, tmp_path):
    """Two different TIFF series selected in the SAME parent directory must have
    distinct identities (§15.5: the old series token carried only the parent
    URI/kind, so two series collided)."""
    parent = tmp_path / "p"
    parent.mkdir()
    first = parent / "alpha_001.tif"
    second = parent / "beta_001.tif"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    _set_signal(widget, inp_type="Image Series", File=str(first))
    token_1 = widget._controls_v2_source_token()
    _set_signal(widget, File=str(second))
    token_2 = widget._controls_v2_source_token()

    assert token_1 is not None
    assert token_1 != token_2


def test_nexus_file_and_entry_yield_distinct_non_none_identities():
    """NeXus file/entry selections must produce distinct, non-None identities
    (§15.5: NeXus candidate edits fell through to ``None``)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _source_selection_identity,
    )

    def identity(nexus_file, entry):
        values = {
            ("NeXus File", "nexus_file"): nexus_file,
            ("NeXus File", "entry"): entry,
        }
        return _source_selection_identity(lambda p: values.get(tuple(p), ""))

    base = identity("/data/a.nxs", "entry1")
    other_file = identity("/data/b.nxs", "entry1")
    other_entry = identity("/data/a.nxs", "entry2")

    assert base is not None
    assert base != other_file
    assert base != other_entry
    assert other_file != other_entry


def test_metadata_format_change_invalidates_observation(widget, tmp_path):
    """A metadata FORMAT change on the same directory invalidates the stored
    motor observation (§15.5: metadata-format/dir changes were unrepresented in
    the token, so a proved observation survived a format switch)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation

    a = tmp_path / "a"
    a.mkdir()
    _set_signal(
        widget, inp_type="Image Directory", img_ext="tif", img_dir=str(a),
        meta_ext="txt",
    )
    widget._controls_v2_gi_motor_observation = GIMotorObservation(
        GIMotorObservation.KNOWN_NONEMPTY, ("halpha",),
        widget._controls_v2_source_token())

    staged = widget.stage_controls_transaction([(("Signal", "meta_ext"), "pdi")])

    assert staged.gi_motor_observation.state == GIMotorObservation.UNKNOWN
    assert "halpha" not in staged.gi_motor_observation.motors


# --------------------------------------------------------------------------
# Item 3 — request-local GI hydration token (§15.11-C tests 5-6)
# --------------------------------------------------------------------------

def test_overlapped_late_a_result_is_rejected(widget, qapp):
    """A starts → B starts → A completes late: A carries its OWN start identity
    and is rejected as superseded (§15.5 A-late-under-B mis-stamp).

    §19.4 update: A's completion now CARRIES A's token (``token=token_a``)
    rather than relying on a FIFO-oldest pop at the completion seam — completion
    order must never select request identity."""
    wrangler = widget.wrangler
    emitted = []
    wrangler.sigGIMotorOptions.connect(emitted.append)
    wrangler.inp_type = "Image Directory"

    wrangler.img_dir = "/tmp/c-source-a"
    fingerprint_a = wrangler._gi_source_fingerprint()
    token_a = wrangler._begin_gi_hydration_request()  # A's request

    wrangler.img_dir = "/tmp/c-source-b"
    wrangler._begin_gi_hydration_request()            # B's request
    # A completes late, under its own token.
    wrangler._emit_gi_hydration(["halpha"], proved=True, token=token_a)
    qapp.processEvents()

    assert emitted[-1].source_fingerprint == fingerprint_a
    assert wrangler.gi_hydration_is_current(emitted[-1]) is False


def test_older_same_source_request_completing_after_newer_is_rejected(widget, qapp):
    """Two requests for the SAME source; the older completes after the newer was
    opened — the older result is rejected on its (lower) generation.

    §19.4 update: the older completion carries the older request's token."""
    wrangler = widget.wrangler
    emitted = []
    wrangler.sigGIMotorOptions.connect(emitted.append)
    wrangler.inp_type = "Image Directory"
    wrangler.img_dir = "/tmp/c-same-source"

    token_older = wrangler._begin_gi_hydration_request()   # older request
    wrangler._begin_gi_hydration_request()                 # newer request
    # The older request completes first, under its own token.
    wrangler._emit_gi_hydration(["halpha"], proved=True, token=token_older)
    qapp.processEvents()

    older = emitted[-1]
    assert wrangler.gi_hydration_is_current(older) is False


def test_targeted_empty_inspection_is_known_empty(widget, qapp):
    """Item-4 partner half (regression guard, NOT a fail-before): a TARGETED
    inspection that proved no motors is KNOWN_EMPTY (resolves to Manual)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation

    wrangler = widget.wrangler
    wrangler.motors = []
    wrangler._gi_motor_knowledge_proved = True        # a targeted inspection ran
    wrangler.set_gi_motor_options()
    qapp.processEvents()

    observation = widget._controls_v2_capture_gi_motor_observation()
    assert observation.state == GIMotorObservation.KNOWN_EMPTY
    assert observation.choices_for_freeze() == ()


# --------------------------------------------------------------------------
# Item 5 — typed source receipt with verified rollback (§15.11-C tests 8-9)
# --------------------------------------------------------------------------

def _configure_container_source(widget, root):
    """Point the source at a container (h5) Image Directory and reconcile a real
    ``DirectoryIndexSession``; return the session."""
    (root / "scan_master.h5").write_bytes(b"x")
    _set_signal(widget, inp_type="Image Directory", img_ext="h5", img_dir=str(root))
    widget._sync_controls_v2_source_index()
    session = widget._controls_v2_source_widget.directory_session
    if session is None or session.configured is None:
        pytest.skip("no real directory session available in this environment")
    return session


def test_real_session_restore_is_verified_exact(widget, tmp_path, monkeypatch):
    """A real DirectoryIndexSession partial reconcile failure whose recovery
    restores the exact prior selection is VERIFIED and reported as a clean
    recovery (no ("Source",) failure)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "scan_master.h5").write_bytes(b"x")
    session = _configure_container_source(widget, a)
    prior = session.configured

    staged = widget.stage_controls_transaction([(("Signal", "img_dir"), str(b))])
    assert staged.source_selection_touched

    real_sync = widget._sync_controls_v2_source_index
    calls = {"n": 0}

    def controlled_sync():
        calls["n"] += 1
        if calls["n"] == 1:
            real_sync()                                # forward: reconcile to B
            raise RuntimeError("partial reconcile failure")
        # recovery: a build-aside owner restores the exact prior selection
        session.configure(
            prior.root, recursive=prior.recursive,
            name_filter=prior.name_filter, suffixes=prior.suffixes)

    monkeypatch.setattr(widget, "_sync_controls_v2_source_index", controlled_sync)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.recovery_failed_path is None
    assert session.configured == prior


def test_nonraising_source_noop_restore_is_recovery_failure(
    widget, tmp_path, monkeypatch
):
    """A non-raising `_sync` that leaves the session pointed at the FAILED
    selection is NOT proof of restoration — the receipt verification catches it
    and reports a ("Source",) recovery failure (§15.6)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "scan_master.h5").write_bytes(b"x")
    session = _configure_container_source(widget, a)

    staged = widget.stage_controls_transaction([(("Signal", "img_dir"), str(b))])
    assert staged.source_selection_touched

    real_sync = widget._sync_controls_v2_source_index
    calls = {"n": 0}

    def controlled_sync():
        calls["n"] += 1
        if calls["n"] == 1:
            real_sync()                                # forward: reconcile to B
            raise RuntimeError("partial reconcile failure")
        # recovery: a non-raising NO-OP that does NOT restore the session

    monkeypatch.setattr(widget, "_sync_controls_v2_source_index", controlled_sync)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.recovery_failed_path == ("Source",)


# --------------------------------------------------------------------------
# Folded §17.7 case-3 — a successful commit clears the refusal latch
# --------------------------------------------------------------------------

def test_successful_correction_clears_refusal_latch(widget):
    """A refused focus-loss edit sets the dedupe latch; a valid correction that
    commits clears it (so a later refusal is a fresh visible message)."""
    widget._controls_v2_last_refusal_signature = None

    widget._on_controls_v2_field_changed(("Int1D", "points"), "4.5")
    assert widget._controls_v2_last_refusal_signature is not None

    widget._on_controls_v2_field_changed(("Int1D", "points"), "500")
    assert widget._controls_v2_last_refusal_signature is None
