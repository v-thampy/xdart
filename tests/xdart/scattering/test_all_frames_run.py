"""All frames leaves the existing Single/Auto Last Run arrival contract intact."""
from pathlib import Path
from threading import Event

from pyqtgraph.Qt import QtTest

from tests.xdart.scattering.test_all_frames_toggle import LEFT, _mode, _on, _toggle
from tests.xdart.scattering.test_e3_join_oracle import _mount, _run, _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.display_context import ContextKind
from xrd_tools.sources.image import TiffSeriesSource


def test_single_run_arrival_follows_auto_last_and_keeps_all_frames_checked(tmp_path, monkeypatch):
    # Pace only real source reads; reduction, publication, Run and painting are production owners.
    entered = {label: Event() for label in (3, 5)}
    release = {label: Event() for label in entered}
    read_path = TiffSeriesSource._read_path

    def paced_read(source, path):
        label = int(Path(path).stem.rsplit("_", 1)[1])
        if label in entered:
            entered[label].set()
            assert release[label].wait(20), "raw-source pacing was not released"
        return read_path(source, path)

    monkeypatch.setattr(TiffSeriesSource, "_read_path", paced_read)
    rig = _mount(monkeypatch, tmp_path, labels=(1, 2, 3, 4, 5), max_display_items=8)
    page, shell = rig.page, rig.shell
    try:
        _mode(page, "Single")
        shell.browser.auto_last.setChecked(True)
        _run(rig)
        _wait(rig.app, lambda: entered[3].is_set() and len(rig.controller.frame_keys) == 2)
        assert rig.lifecycle.phase is RunPhase.RUNNING
        _toggle(page)
        assert _on(page) and len(rig.controller.navigation.selected) == 2
        release[3].set()
        _wait(rig.app, lambda: entered[5].is_set() and len(rig.controller.frame_keys) == 4)

        def latest_is_painted():
            nav = rig.controller.navigation
            latest = nav.frames[-1]
            return (nav.current is latest and nav.selected == (latest,)
                    and shell.scientific.trace_history_keys == (latest,))

        _wait(rig.app, latest_is_painted, diagnostic=lambda: str(rig.controller.navigation))
        assert rig.controller.selection.kind is ContextKind.ACQUISITION
        assert _on(page) and page._processed_browser.auto_last
        # Auto Last itself also keeps the persisted preference while selecting just the latest.
        QtTest.QTest.mouseClick(shell.browser.auto_last, LEFT)
        QtTest.QTest.mouseClick(shell.browser.auto_last, LEFT)
        assert _on(page) and len(rig.controller.navigation.selected) == 1
    finally:
        for gate in release.values():
            gate.set()
        assert rig.close().cleanup_status is CleanupStatus.CLEANED
        page.deleteLater()
        rig.app.processEvents()
