from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from tests.xdart.test_o3n_execution_owner import (  # noqa: E402
    _select_nexus,
    _start_recorder,
    _write_poni,
    qapp,
    widget,
)


def _arm_entry_only(wrangler, tmp_path: Path, *, xye_only: bool = False,
                    runnable: bool = False):
    """Arm a container whose ``entry`` group exists but carries NO raw stack.

    ``runnable=True`` gives it one, for the rows whose subject is the output
    transaction rather than the refusal (§16.4: output ownership is unreachable
    until the source is proved runnable, so those rows need a real source).
    """
    import numpy as np

    source = tmp_path / "raw" / "acq.nxs"
    source.parent.mkdir()
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        if runnable:
            detector = entry.create_group("instrument/detector")
            detector.create_dataset(
                "data", data=np.zeros((2, 8, 8), dtype=np.float32))
    output = tmp_path / "out"
    output.mkdir()
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni")
    )
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(output))
    wrangler.parameters.child("Project", "project_folder").setValue(str(tmp_path))
    if xye_only:
        wrangler.processingModeCombo.setCurrentText("Int 1D (XYE)")
    return source, output


def test_overwrite_does_not_destroy_prior_target_before_source_is_runnable(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    source, output = _arm_entry_only(wrangler, tmp_path)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    target = output / f"{source.stem}.nxs"
    prior = b"PRIOR-DURABLE-RESULT"
    target.write_bytes(prior)

    thread.run()

    assert target.read_bytes() == prior


def test_xye_only_overwrite_does_not_delete_an_nxs_it_will_never_replace(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    # The subject here is the XYE-only OUTPUT TRANSACTION, so the source must be
    # runnable; the sibling row above owns the not-runnable refusal.
    source, output = _arm_entry_only(
        wrangler, tmp_path, xye_only=True, runnable=True)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    assert thread.run_configuration.run_options["xye_only"] is True
    target = output / f"{source.stem}.nxs"
    prior = b"PRIOR-DURABLE-RESULT"
    target.write_bytes(prior)

    thread._initialize_scan(source.stem)

    assert target.read_bytes() == prior


def test_append_existing_different_identity_is_refused_before_writer(
    tmp_path, monkeypatch
):
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import nexusThread
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.session import RunIntent
    from xrd_tools.session.run_configuration import RunConfigurationRefused

    output = tmp_path / "same.nxs"
    output.write_bytes(b"prior identity")
    source = tmp_path / "other" / "same.nxs"
    source.parent.mkdir()
    source.write_bytes(b"source")
    frozen = RunIntent(
        output_mode="Append",
        save_path=str(tmp_path),
        source_spec=SourceSpec(source, SourceKind.NEXUS_STACK, entry="entry"),
    ).freeze()
    writes = []
    scan = SimpleNamespace(
        data_file=str(output),
        _save_to_nexus=lambda: writes.append("writer reached"),
    )
    # O-3N.R.2 §17.4: the run's output transaction lives on the prepared
    # envelope, so the qualification is asked of a REAL envelope over a REAL
    # target rather than of a latch on a stand-in worker.
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        PreparedNexusExecution,
    )

    prepared = PreparedNexusExecution(
        frozen, nexusThread._frozen_source_target(frozen))
    worker = SimpleNamespace(
        _execution=prepared,
        file_lock=__import__("threading").RLock(),
    )
    monkeypatch.setattr(
        "xdart.gui.tabs.static_scan.wranglers.wrangler_widget._get_h5pool",
        lambda: SimpleNamespace(pause=lambda _p: None, resume=lambda _p: None),
    )

    with pytest.raises(RunConfigurationRefused):
        nexusThread._prepare_output_for_run(worker, prepared, scan)

    assert writes == []
    assert prepared.append_qualified is False
