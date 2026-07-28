"""O-3N.R — the reviewer's four-case depth oracle (handoff §15.5 R0).

Promoted from ``/Users/vthampy/repos/tmp/test_codex_o3n_depth.py``, where it is
``4 failed`` at parent ``30584d5b``.  It states the three §15 P1 defects:

* blank Entry executed as ``"entry"`` while the frozen identity and provenance
  claimed ``None`` -- one accepted identity with two meanings;
* ``_initialize_scan()`` never installed the accepted output on
  ``scan.data_file``, so the real writer targeted whatever the display scan
  happened to point at and correctness depended on the async GUI
  ``sigUpdateFile`` chain;
* a source and a NeXus Output directory that resolve to the SAME file were
  admitted and executed -- the F-NXS-1 raw-acquisition-overwrite hazard, newly
  reachable because O-3N made NeXus runs executable.

ONE ROW RESHAPED, deliberately and visibly.  The preserved
``test_execution_equivalent_entry_spellings_have_one_identity`` asserted that a
blank Entry and an explicit ``"entry"`` must produce the same
``source.entry`` and the same fingerprint.  Its own docstring states the
premise: *"If blank means ``entry`` at execution..."*.  §15.2's ratified ruling
removes that premise -- "refuse blank entry before carrier publication; do not
normalize an operator-cleared field back into a value and do not retain a
worker-only fallback" -- and §15.5 R0's first bullet requires blank Entry to be
"a typed, zero-delta refusal ... never invented later".  The preserved
assertion and the ratified ruling cannot both hold: normalizing blank to
``"entry"`` to equalize the fingerprints is exactly what §15.2 forbids, and
``test_codex_o3n_malformed.py`` independently requires ``frozen.source.entry ==
""`` for a cleared editor.  The row is therefore promoted as the ruling states
it: blank never becomes an execution identity at all, so it can never be a
second spelling of one.  Flagged in the handback for the reviewer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from tests.xdart.test_o3n_nexus_freeze_identity import _write_poni  # noqa: E402
from xrd_tools.session.run_configuration import (  # noqa: E402
    RunConfigurationRefused,
)


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
        value._acquisition_context = None
        value._browse_context = None
        value._display_selection = None
        value._run_active = False
        try:
            value._controls_v2_refresh_timer.cancel()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _select_nexus(widget):
    from xdart.gui.tabs.static_scan.wranglers import nexusWrangler

    stack = widget.ui.wranglerStack
    for index in range(stack.count()):
        page = stack.widget(index)
        if isinstance(page, nexusWrangler):
            stack.setCurrentIndex(index)
            widget.set_wrangler(index)
            return page
    raise AssertionError("missing NeXus wrangler")


def _arm(wrangler, tmp_path, *, entry):
    src = tmp_path / "source.nxs"
    src.write_bytes(b"source")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    # O-3N.R.1 §16.5: a constructible calibration is now an ADMISSION fact, so
    # every row that expects an accepted Start must arm one.  The preserved
    # module armed none, which under the corrected rule is itself a refusal --
    # pinned by test_o3nr1_totality::test_public_start_without_calibration_
    # refuses_zero_delta.
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue(entry)
    wrangler.parameters.child("Output", "h5_dir").setValue(str(out))
    return src, out


def test_blank_entry_execution_and_provenance_are_one_fact(widget, tmp_path):
    """One accepted identity may not claim None while executing ``entry``."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    _arm(wrangler, tmp_path, entry="")
    frozen = widget._prepare_controls_v2_run_configuration()
    target = nexusThread._frozen_source_target(frozen)

    assert frozen.source.entry == target.entry


def test_execution_equivalent_entry_spellings_have_one_identity(
        widget, tmp_path):
    """Reshaped per §15.2 (see the module docstring): a cleared Entry never
    becomes an execution identity, so it can never be a second spelling of the
    explicit one.  The refusal is typed and happens at admission."""
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )

    wrangler = _select_nexus(widget)
    _arm(wrangler, tmp_path, entry="")
    blank = widget._prepare_controls_v2_run_configuration()
    assert blank.source.entry == "", "a cleared Entry was invented into a value"
    with pytest.raises(RunConfigurationRefused):
        wranglerWidget._admit_run_configuration(wrangler, "nexus-start")

    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    explicit = widget._prepare_controls_v2_run_configuration()
    assert explicit.source.entry == "entry"
    assert explicit.fingerprint != blank.fingerprint, (
        "a refused blank Entry must not share the accepted identity")
    wranglerWidget._admit_run_configuration(wrangler, "nexus-start")
    assert wrangler.run_configuration.source.entry == "entry"


def test_worker_scan_writes_the_frozen_output_target(
        widget, tmp_path, monkeypatch):
    """The writer consumes ``scan.data_file``, so it must equal frozen output."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out = _arm(wrangler, tmp_path, entry="entry")
    monkeypatch.setattr(nexusThread, "start", lambda self: None)
    wrangler.start()
    thread = wrangler.thread
    expected = nexusThread._frozen_source_target(
        thread.run_configuration).output_path
    scan = thread._initialize_scan(src.stem)

    assert thread.fname == expected
    assert scan.data_file == expected


def test_same_file_input_output_is_refused_before_worker_start(
        widget, tmp_path, monkeypatch):
    """A newly executable NeXus run may not overwrite its raw acquisition."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src = tmp_path / "source.nxs"
    src.write_bytes(b"source")
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(tmp_path))
    started = []
    monkeypatch.setattr(nexusThread, "start",
                        lambda self: started.append(self))

    wrangler.start()

    assert started == []
