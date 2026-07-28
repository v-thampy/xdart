"""O-3N.R — the NeXus run-scoped execution owner (handoff §15.4, §15.5 R1/R2).

§15.1-§15.3 are three symptoms of one root cause §15.4 names: the newly
reachable NeXus worker still took scientific and output decisions from
post-admission mutable state.  These rows pin the closure — one immutable
execution target derived from the accepted ``FrozenRunConfiguration``, a
worker-owned preflight before any source-content or writer I/O, and a
worker-owned run scan whose ``data_file`` IS the frozen output before any
signal, periodic save, final save or XYE flush.

Production-wired: the real ``staticWidget``, the real NeXus page from the real
stack, the real replacement ``nexusThread``, the real headless collision owner,
real HDF5 fixtures written with h5py, and the real writer/provenance reader.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from tests.xdart.test_o3n_nexus_freeze_identity import (  # noqa: E402,F401
    _PONI,
    _select_nexus,
    _start_recorder,
    _write_poni,
    qapp,
    widget,
)


# --------------------------------------------------------------------------- #
# Real HDF5 fixtures
# --------------------------------------------------------------------------- #

def _write_nexus(path, entry="entry", frames=2, shape=(8, 8)):
    """A real, readable NeXus container with ONE named NXentry."""
    path = Path(path)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(entry)
        group.attrs["NX_class"] = "NXentry"
        detector = group.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset(
            "data",
            data=np.arange(frames * shape[0] * shape[1],
                           dtype=np.float32).reshape((frames,) + shape))
    return path


def _arm_real(wrangler, tmp_path, *, entry="entry", selected=None,
              out_name="out", source_name="acq.nxs"):
    """Arm the NeXus page against a REAL container."""
    src = _write_nexus(tmp_path / source_name, entry=entry)
    out = tmp_path / out_name
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue(
        entry if selected is None else selected)
    wrangler.parameters.child("Output", "h5_dir").setValue(str(out))
    wrangler.parameters.child("Project", "project_folder").setValue(
        str(tmp_path))
    return src, out


def _started(wrangler, tmp_path, monkeypatch, **kw):
    """Drive the REAL Start to an admitted run without launching the worker."""
    src, out = _arm_real(wrangler, tmp_path, **kw)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    return src, out, wrangler.thread


# --------------------------------------------------------------------------- #
# §15.2 — strict entry: the selected entry is the one opened, or refuse
# --------------------------------------------------------------------------- #

def test_strict_entry_preflight_refuses_a_missing_entry(
        widget, tmp_path, monkeypatch):
    """The shared reader falls back to the first NXentry.  Execution may not:
    provenance would keep claiming the entry the operator selected."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )
    from xrd_tools.session.run_configuration import RunConfigurationRefused

    wrangler = _select_nexus(widget)
    src, out, thread = _started(
        wrangler, tmp_path, monkeypatch, entry="actual", selected="missing")
    target = nexusThread._frozen_source_target(thread.run_configuration)
    assert target.entry == "missing"

    with pytest.raises(RunConfigurationRefused):
        nexusThread._preflight_execution_target(thread, target)

    assert list(out.iterdir()) == [], "the refusal wrote output anyway"


def test_strict_entry_preflight_admits_the_selected_entry(
        widget, tmp_path, monkeypatch):
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out, thread = _started(
        wrangler, tmp_path, monkeypatch, entry="scan_7", selected="scan_7")
    target = nexusThread._frozen_source_target(thread.run_configuration)

    nexusThread._preflight_execution_target(thread, target)   # must not raise

    assert target.entry == "scan_7"
    assert target.uri == str(src)


def test_the_worker_run_entry_refuses_a_missing_entry_before_any_output(
        widget, tmp_path, monkeypatch):
    """The refusal is on the REAL worker entry, before the reduction body."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out, thread = _started(
        wrangler, tmp_path, monkeypatch, entry="actual", selected="missing")
    reached = []
    monkeypatch.setattr(nexusThread, "_run_impl",
                        lambda self, frozen: reached.append(True))

    thread.run()

    assert reached == [], "the worker reduced against a fallback entry"
    assert list(out.iterdir()) == []


# --------------------------------------------------------------------------- #
# §15.3 — collision identity is filesystem identity, not spelling
# --------------------------------------------------------------------------- #

def test_link_identity_collision_is_refused_not_lexical(
        widget, tmp_path, monkeypatch):
    """A symlinked output directory spells differently and IS the source dir."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    src = _write_nexus(raw / "acq.nxs")
    link = tmp_path / "linked-out"
    os.symlink(raw, link)

    wrangler = _select_nexus(widget)
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(link))
    started = _start_recorder(wrangler, monkeypatch)
    before = src.read_bytes()

    wrangler.start()

    assert started == [], (
        "a symlinked output directory bypassed the collision guard")
    assert src.read_bytes() == before, "the raw acquisition was touched"


# --------------------------------------------------------------------------- #
# §15.4 — the execution target owns science and output policy
# --------------------------------------------------------------------------- #

def test_execution_poni_comes_from_frozen_values_not_the_panel(
        widget, tmp_path, monkeypatch):
    """`setup()` used to reread the PONI editor and rebuild `self.poni`."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    accepted = dict(frozen.poni_values or {})
    assert accepted, "the accepted configuration carries no PONI values"

    poisoned = tmp_path / "poison.poni"
    poisoned.write_text(_PONI.replace("Distance: 0.1", "Distance: 9.9"))
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        str(poisoned))
    wrangler.poni_file = str(poisoned)

    poni = nexusThread._execution_poni(thread, frozen)

    assert poni is not None
    assert poni.to_dict()["dist"] == pytest.approx(accepted["dist"])
    assert poni.to_dict()["dist"] != pytest.approx(9.9)


def test_source_base_comes_from_the_frozen_project_root(
        widget, tmp_path, monkeypatch):
    """Portable @source_base provenance must not follow a post-admission edit."""
    wrangler = _select_nexus(widget)
    _arm_real(wrangler, tmp_path)
    _start_recorder(wrangler, monkeypatch)
    poison_root = tmp_path / "poison-root"
    poison_root.mkdir()
    original = type(widget)._apply_controls_v2_run_state

    def _poison_between_admission_and_setup(self, frozen):
        # The real ordering point: after admission, before wrangler.setup().
        result = original(self, frozen)
        wrangler.parameters.child("Project", "project_folder").setValue(
            str(poison_root))
        wrangler.project_folder = str(poison_root)
        return result

    monkeypatch.setattr(type(widget), "_apply_controls_v2_run_state",
                        _poison_between_admission_and_setup)

    wrangler.start()
    thread = wrangler.thread
    frozen = thread.run_configuration
    assert frozen.project_root == str(tmp_path)

    scan = thread._initialize_scan(Path(frozen.source.uri).stem)

    assert str(scan.source_base or "") == str(tmp_path)
    assert str(poison_root) not in str(scan.source_base or "")


def test_no_gui_thread_hdf5_entry_inspection_after_admission(
        widget, tmp_path, monkeypatch):
    """§15.4 item 3: `_emit_gi_motor_options()` used to reread Entry and open
    the container synchronously on the GUI thread inside `setup()`."""
    import xdart.gui.tabs.static_scan.wranglers.nexus_wrangler as nw

    wrangler = _select_nexus(widget)
    _arm_real(wrangler, tmp_path)
    _start_recorder(wrangler, monkeypatch)
    opened = []
    monkeypatch.setattr(nw, "read_nexus",
                        lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(
                            AssertionError("GUI-thread HDF5 read after admission")))

    wrangler.start()

    assert opened == []


def test_the_execution_target_carries_the_accepted_output_mode(
        widget, tmp_path, monkeypatch):
    """§15.4 item 4: the final save hard-coded ``replace=False``."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )

    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    target = nexusThread._frozen_source_target(thread.run_configuration)

    assert target.output_mode == "Overwrite"
    assert thread.run_configuration.output_mode == "Overwrite"


def test_overwrite_replaces_a_prior_same_stem_result(
        widget, tmp_path, monkeypatch):
    """An Overwrite run's FIRST writer action replaces; it may not append rows
    from a different source/entry identity under new provenance."""
    from xrd_tools.core.provenance import read_provenance

    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    scan = thread._initialize_scan(Path(frozen.source.uri).stem)
    stale = Path(scan.data_file)
    stale.write_bytes(b"not a real nxs")

    thread._prepare_output_for_run(frozen, scan)
    scan.save_to_nexus()

    stored = (read_provenance(str(stale)).get("config") or {}).get(
        "run_configuration")
    assert stored is not None
    assert (int(stored["generation"]), stored["fingerprint"]) == frozen.identity
    assert stored["source"]["uri"] == str(src)


# --------------------------------------------------------------------------- #
# §15.5 R2 — the writer is independent of the GUI, and lands only under
# the frozen save path
# --------------------------------------------------------------------------- #

def test_writer_target_is_independent_of_the_gui_sigupdatefile_delivery(
        widget, tmp_path, monkeypatch):
    """Hold the GUI consumer: worker correctness may not depend on it."""
    wrangler = _select_nexus(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    held = []
    try:
        wrangler.sigUpdateFile.disconnect()
    except (RuntimeError, TypeError):
        pass
    wrangler.sigUpdateFile.connect(lambda *a: held.append(a))
    frozen = thread.run_configuration

    scan = thread._initialize_scan(Path(frozen.source.uri).stem)

    assert scan.data_file == str(out / f"{Path(frozen.source.uri).stem}.nxs")
    assert held == [], "the row did not actually hold the GUI consumer"


def test_xye_output_lands_only_under_the_frozen_save_path(
        widget, tmp_path, monkeypatch):
    """§15.4 item 5: XYE paths derive from ``scan.data_file``."""
    wrangler = _select_nexus(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    scan = thread._initialize_scan(Path(frozen.source.uri).stem)

    xye_root = Path(os.path.dirname(scan.data_file)) / scan.name

    assert str(xye_root).startswith(str(out))
    assert not str(xye_root).startswith(str(tmp_path / "xdart_processed_data"))


def test_one_identity_reaches_the_real_writer_without_a_manual_repoint(
        widget, tmp_path, monkeypatch):
    """§15.5 R2: no test-side ``scan.data_file = thread.fname``.  The worker's
    own ``_initialize_scan`` must have installed the frozen target already."""
    from xrd_tools.core.provenance import read_provenance

    wrangler = _select_nexus(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration

    scan = thread._initialize_scan(Path(frozen.source.uri).stem)
    scan.save_to_nexus()

    written = Path(scan.data_file)
    assert written.parent == out, "output landed outside the frozen save path"
    stored = (read_provenance(str(written)).get("config") or {}).get(
        "run_configuration")
    assert stored is not None
    assert (int(stored["generation"]), stored["fingerprint"]) == frozen.identity
    assert stored["source"]["uri"] == str(src)
    assert stored["source"]["entry"] == frozen.source.entry
    assert stored["save_path"] == str(out)
