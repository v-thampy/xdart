"""O-3N — the NeXus page's source, entry and output are FROZEN, not re-read.

Frozen acceptance oracle for handoff Section 14, written BEFORE any production
edit and red at parent ``2d043774``.

Section 14.1 named one namespace/ownership defect.  ``_controls_v2_freeze_source_spec()``
reads only the image page's ``Signal/inp_type`` + ``Signal/File``, and
``_prepare_controls_v2_run_configuration()`` reads only ``Project/h5_dir`` -- but
the NeXus page owns ``NeXus File/nexus_file``, ``NeXus File/entry`` and
``Output/h5_dir``.  Every NeXus freeze therefore carried ``source=None`` and
``save_path=""``, ``nexusWrangler`` declared neither as required, and
``setup()``/``_run_impl()`` re-read file, entry and output from the mutable Qt
tree AFTER admission.  A source-only patch would turn the O-3 guard green while
leaving W-1's freeze-once execution contract false, so the rows below pin the
whole chain.

Production-wired throughout (CLAUDE.md rule 2): the real ``staticWidget``, the
real ``nexusWrangler`` taken from the real wrangler stack, the real
``nexusThread`` the replacement ``setup()`` builds, the real admission owner and
the real writer/provenance reader.

Row map (Section 14.3):

A  a real NeXus Start freezes NEXUS_STACK, the exact URI, the exact entry and
   the exact output directory
B  changing only the file, or only the entry, changes the fingerprint at the
   SAME intent generation
C  a blank source and a blank output each refuse before pending, carrier,
   session, button or worker mutation
D  poisoning file/entry/output after admission cannot change what the
   replacement worker opens or where it writes
E  one exact identity reaches wrapper, thread, worker scan, written provenance
   and reload
F  O-3's acquisition source is exactly ``frozen.source.uri``
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from xrd_tools.core.scan import SourceKind  # noqa: E402
from xrd_tools.session.run_configuration import (  # noqa: E402
    RunConfigurationRefused,
)


# --------------------------------------------------------------------------- #
# Fixtures
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
    """Take the REAL NeXus page out of the REAL wrangler stack."""
    from xdart.gui.tabs.static_scan.wranglers import nexusWrangler

    stack = widget.ui.wranglerStack
    for index in range(stack.count()):
        page = stack.widget(index)
        if isinstance(page, nexusWrangler):
            stack.setCurrentIndex(index)
            widget.set_wrangler(index)
            return page
    raise AssertionError("missing NeXus wrangler")


_PONI = """\
poni_version: 2
Detector: Pilatus100k
Detector_config: {}
Distance: 0.1
Poni1: 0.02
Poni2: 0.02
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1e-10
"""


def _write_poni(path: Path) -> str:
    path.write_text(_PONI)
    return str(path)


def _arm_nexus(wrangler, tmp_path, *, name="source.nxs", entry="entry",
               source=True, output=True):
    """Arm exactly what the NeXus page owns: file, entry, calibration, output."""
    src = tmp_path / name
    src.write_bytes(b"source")
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(
        str(src) if source else "")
    wrangler.parameters.child("NeXus File", "entry").setValue(entry)
    wrangler.parameters.child("Output", "h5_dir").setValue(
        str(out) if output else "")
    return src, out


def _start_recorder(wrangler, monkeypatch):
    """Capture every observable a REFUSED Start must not touch."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    events = []
    monkeypatch.setattr(nexusThread, "start",
                        lambda self: events.append("worker-start"))
    monkeypatch.setattr(type(wrangler), "_save_to_session",
                        lambda self: events.append("session"))
    wrangler.sigStart.connect(lambda: events.append("sigStart"))
    return events


def _zero_delta_snapshot(widget, wrangler):
    thread = wrangler.thread
    return {
        "command": getattr(wrangler, "command", None),
        "thread": thread,
        "thread_command": getattr(thread, "command", None),
        "carrier": getattr(wrangler, "run_configuration", None),
        "ledger": getattr(wrangler, "_admitted_run_configuration", None),
        "thread_carrier": getattr(thread, "run_configuration", None),
        "thread_ledger": getattr(thread, "_admitted_run_configuration", None),
        "source_spec": getattr(wrangler, "source_spec", None),
        "start_enabled": bool(wrangler.startButton.isEnabled()),
        "stop_enabled": bool(wrangler.stopButton.isEnabled()),
        "pending": getattr(
            widget, "_pending_controls_v2_run_configuration", None),
        "floor": int(getattr(wrangler, "run_configuration_floor", 0) or 0),
    }


# --------------------------------------------------------------------------- #
# A — the freeze names source, entry and output
# --------------------------------------------------------------------------- #

def test_real_nexus_freeze_names_source_entry_and_output(widget, tmp_path):
    """Section 14.3 row 1, and the reviewer's preserved parent-red probe."""
    wrangler = _select_nexus(widget)
    src, out = _arm_nexus(wrangler, tmp_path, entry="entry/data")

    spec = widget._controls_v2_freeze_source_spec()
    frozen = widget._prepare_controls_v2_run_configuration()

    assert spec is not None, "the NeXus page still freezes no typed source"
    assert str(spec.uri) == str(src)
    assert spec.kind is SourceKind.NEXUS_STACK
    assert spec.entry == "entry/data"

    assert frozen.source is not None
    assert frozen.source.uri == str(src)
    assert frozen.source.source_kind == "nexus_stack"
    assert frozen.source.entry == "entry/data"
    assert frozen.save_path == str(out)


def test_the_image_page_source_derivation_is_unchanged(widget, tmp_path):
    """The bounded NeXus branch must not touch the image namespace."""
    wrangler = widget.wrangler                       # the default image page
    assert type(wrangler).__name__ == "imageWrangler"
    raw = tmp_path / "frame_0001.tif"
    raw.write_bytes(b"")
    out = tmp_path / "img_out"
    out.mkdir()
    signal = wrangler.parameters.child("Signal")
    signal.child("inp_type").setValue("Image Series")
    signal.child("File").setValue(str(raw))
    wrangler.parameters.child("Project", "h5_dir").setValue(str(out))

    frozen = widget._prepare_controls_v2_run_configuration()

    assert frozen.source is not None
    assert frozen.source.source_kind == "tiff_series"
    assert frozen.save_path == str(out), (
        "the image page's output must keep coming from Project/h5_dir")


def test_the_staged_candidate_derives_the_same_nexus_source(widget, tmp_path):
    """Section 14.2 item 1: live freeze and staged candidate share ONE
    derivation, so a file/entry edit moves the same fingerprint."""
    wrangler = _select_nexus(widget)
    src, _ = _arm_nexus(wrangler, tmp_path, entry="entry/data")

    live = widget._controls_v2_freeze_source_spec()
    # the REAL staged candidate: its getter reads the committed snapshot of the
    # bound source-selection paths, which already include the NeXus pair.
    candidate = widget._controls_v2_candidate_source_spec(
        widget._controls_v2_new_stage_candidate())

    assert candidate is not None, "the candidate path has no NeXus branch"
    assert (str(candidate.uri), candidate.entry, candidate.kind) == (
        str(live.uri), live.entry, live.kind)
    assert str(candidate.uri) == str(src)


# --------------------------------------------------------------------------- #
# B — file and entry participate in the identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,first,second", [
    ("nexus_file", "a.nxs", "b.nxs"),
    ("entry", "entry", "entry/other"),
])
def test_editing_only_one_nexus_field_moves_the_fingerprint(
        widget, tmp_path, field, first, second):
    """Section 14.3 row 2.  Two real selections at the SAME intent generation
    must not produce the same identity."""
    wrangler = _select_nexus(widget)
    if field == "nexus_file":
        _arm_nexus(wrangler, tmp_path, name=first)
        before = widget._prepare_controls_v2_run_configuration()
        other = tmp_path / second
        other.write_bytes(b"source")
        wrangler.parameters.child("NeXus File", "nexus_file").setValue(
            str(other))
    else:
        _arm_nexus(wrangler, tmp_path, entry=first)
        before = widget._prepare_controls_v2_run_configuration()
        wrangler.parameters.child("NeXus File", "entry").setValue(second)
    after = widget._prepare_controls_v2_run_configuration()

    assert int(after.generation) == int(before.generation), (
        "the rows must compare identities at ONE generation")
    assert after.fingerprint != before.fingerprint, (
        f"{field} does not participate in the run identity")


# --------------------------------------------------------------------------- #
# C — blank source / blank output refuse with zero delta
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("missing", ["source", "output"])
def test_incomplete_nexus_input_refuses_before_any_mutation(
        widget, tmp_path, monkeypatch, missing):
    """Section 14.2 item 3 / Section 14.3 row 3.  Blank input or output is a
    typed, visible, zero-delta refusal."""
    wrangler = _select_nexus(widget)
    _arm_nexus(wrangler, tmp_path,
               source=(missing != "source"), output=(missing != "output"))
    events = _start_recorder(wrangler, monkeypatch)
    said = []
    monkeypatch.setattr(type(wrangler), "_set_status_text",
                        lambda self, text: said.append(str(text)))
    before = _zero_delta_snapshot(widget, wrangler)

    wrangler.start()

    assert events == [], f"a refused NeXus Start still did {events}"
    assert _zero_delta_snapshot(widget, wrangler) == before, (
        "the refusal was not zero-delta")
    assert said, "the refusal was silent"


@pytest.mark.parametrize("missing", ["source", "output"])
def test_incomplete_nexus_input_is_a_typed_admission_refusal(
        widget, tmp_path, missing):
    """The refusal is the shared typed one, raised by the admission owner."""
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        wranglerWidget,
    )

    wrangler = _select_nexus(widget)
    _arm_nexus(wrangler, tmp_path,
               source=(missing != "source"), output=(missing != "output"))

    with pytest.raises(RunConfigurationRefused) as caught:
        wranglerWidget._admit_run_configuration(wrangler, "nexus-start")

    assert missing.replace("output", "save_path") in str(caught.value)


def test_the_nexus_wrangler_declares_both_required_frozen_values(widget):
    """Section 14.2 item 3, as a fact about the production object."""
    wrangler = _select_nexus(widget)
    assert set(getattr(
        wrangler, "_admission_required_frozen_values", ())) == {
            "save_path", "source"}


# --------------------------------------------------------------------------- #
# D — post-admission poisoning cannot move input or output
# --------------------------------------------------------------------------- #

def test_setup_initializes_the_replacement_worker_from_the_frozen_values(
        widget, tmp_path, monkeypatch):
    """Section 14.2 item 4.  ``setup()`` REPLACES the thread; the new worker's
    input and output must come from the admitted object."""
    wrangler = _select_nexus(widget)
    src, out = _arm_nexus(wrangler, tmp_path, entry="entry/data")
    _start_recorder(wrangler, monkeypatch)

    wrangler.start()

    frozen = wrangler.run_configuration
    thread = wrangler.thread
    assert frozen is not None and frozen.source is not None
    assert thread.nexus_file == frozen.source.uri
    assert thread.entry == frozen.source.entry
    assert thread.fname == os.path.join(
        frozen.save_path, f"{Path(frozen.source.uri).stem}.nxs")
    assert str(out) == frozen.save_path
    assert str(src) == frozen.source.uri


def test_poisoning_qt_and_mirrors_cannot_move_the_worker_input_or_output(
        widget, tmp_path, monkeypatch):
    """Section 14.3 row 4.  After admission the answer comes from the accepted
    object, so poisoning every mutable mirror changes nothing."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out = _arm_nexus(wrangler, tmp_path, entry="entry/data")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    frozen = wrangler.run_configuration
    thread = wrangler.thread
    expected = nexusThread._frozen_source_target(frozen)

    poison_dir = tmp_path / "poison"
    poison_dir.mkdir()
    poison_file = poison_dir / "poison.nxs"
    poison_file.write_bytes(b"poison")
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(
        str(poison_file))
    wrangler.parameters.child("NeXus File", "entry").setValue("poison/entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(poison_dir))
    wrangler.nexus_file = str(poison_file)
    wrangler.entry = "poison/entry"
    wrangler.h5_dir = str(poison_dir)
    wrangler.fname = str(poison_dir / "poison.nxs")
    thread.nexus_file = str(poison_file)
    thread.entry = "poison/entry"
    thread.fname = str(poison_dir / "poison.nxs")

    observed = nexusThread._frozen_source_target(frozen)

    assert observed == expected
    assert observed.uri == str(src)
    assert observed.entry == "entry/data"
    assert observed.output_path == os.path.join(str(out), f"{src.stem}.nxs")
    assert str(poison_dir) not in observed.output_path


def test_the_worker_run_entry_reinitializes_its_cursors_from_the_frozen_object(
        widget, tmp_path, monkeypatch):
    """Runtime cursor state may stay mutable, but it is INITIALIZED FROM and
    qualified by the accepted values -- a poisoned mirror is overwritten before
    anything opens."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out = _arm_nexus(wrangler, tmp_path, entry="entry/data")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    frozen = wrangler.run_configuration
    thread = wrangler.thread

    thread.nexus_file = str(tmp_path / "poison.nxs")
    thread.entry = "poison/entry"
    thread.fname = str(tmp_path / "poison" / "poison.nxs")

    nexusThread._adopt_frozen_source_target(thread, frozen)

    assert thread.nexus_file == str(src)
    assert thread.entry == "entry/data"
    assert thread.fname == os.path.join(str(out), f"{src.stem}.nxs")
    assert thread.scan_name == src.stem


# --------------------------------------------------------------------------- #
# E — one identity through wrapper, thread, worker scan, provenance, reload
# --------------------------------------------------------------------------- #

def test_one_nexus_identity_from_start_through_provenance_and_reload(
        widget, tmp_path, monkeypatch):
    """Section 14.3 row 5.  The wrapper, the replacement worker, the scan the
    worker initializes, the persisted provenance and a real reload all carry the
    same object/identity AND the same source URI + entry."""
    from xrd_tools.core.provenance import read_provenance

    wrangler = _select_nexus(widget)
    src, out = _arm_nexus(wrangler, tmp_path, entry="entry/data")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()

    frozen = wrangler.run_configuration
    thread = wrangler.thread
    hops = {"wrapper": frozen,
            "wrapper_ledger": wrangler._admitted_run_configuration,
            "thread": thread.run_configuration,
            "thread_ledger": thread._admitted_run_configuration}
    scan = thread._initialize_scan(Path(frozen.source.uri).stem)
    hops["worker_scan"] = scan.run_configuration
    for name, value in hops.items():
        assert value is frozen, f"hop {name} carried a different object"

    identity = frozen.identity
    assert (int(scan.run_configuration_generation),
            scan.run_configuration_fingerprint) == identity
    provenance = scan.run_configuration_provenance
    assert provenance["source"]["uri"] == str(src)
    assert provenance["source"]["entry"] == "entry/data"
    assert provenance["save_path"] == str(out)

    # The GUI's ``sigUpdateFile`` hop is what points the display scan at the
    # frozen-derived output; the worker's own contribution is ``thread.fname``,
    # asserted against the frozen values above and here.
    scan.data_file = thread.fname
    scan.save_to_nexus()
    stored = (read_provenance(thread.fname).get("config") or {}).get(
        "run_configuration")
    assert stored is not None, "the run identity was not persisted"
    assert (int(stored["generation"]), stored["fingerprint"]) == identity
    assert stored["source"]["uri"] == str(src)
    assert stored["source"]["entry"] == "entry/data", (
        "the reloaded provenance cannot name the entry that was reduced")


# --------------------------------------------------------------------------- #
# F — O-3's acquisition source is exactly frozen.source.uri
# --------------------------------------------------------------------------- #

def test_the_o3_acquisition_context_source_is_the_frozen_nexus_uri(
        widget, tmp_path, monkeypatch):
    """Section 14.3 row 6.  The O-3 source guard stays unchanged; a corrected
    NeXus admission is what makes it pass, and the acquisition owner's source is
    the accepted URI -- never the output path or a mutable display mirror."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        RUN_ORIGIN_WRANGLER,
    )

    wrangler = _select_nexus(widget)
    src, _ = _arm_nexus(wrangler, tmp_path, entry="entry/data")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    frozen = wrangler.run_configuration

    widget._enter_run_state(origin=RUN_ORIGIN_WRANGLER,
                            run_configuration=frozen)
    context = widget._acquisition_context

    assert context is not None, "the accepted NeXus run installed no context"
    assert context.admitted_source == str(src)
    assert context.hydration_owner.source == str(src)
    assert context.run_configuration is frozen
