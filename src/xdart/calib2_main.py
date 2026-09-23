"""Standalone pyFAI calibration with save reporting and 2026.5.0 corrections.

The PONI builder below is derived from pyFAI's IntegrationTask (MIT, ESRF;
see LICENSE-pyFAI-backport). Its only change is to preserve an explicit False
when a sensor is selected with parallax disabled. Remove with the qualified
upstream replacement; this is not a second calibration implementation.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


CALIBRATION_SAVE_REPORT_ENV = "XDART_CALIBRATION_SAVE_REPORT"


@contextmanager
def capture_poni_saves(report_path):
    """Report the last successful upstream GUI save, never directory mtimes.

    pyFAI's save method synchronizes its filename model only after the file
    has closed successfully. Observe that signal only while a Save is active;
    merely loading a PONI or cancelling Save As must not count as a save.
    """
    from pyFAI.gui.tasks.IntegrationTask import IntegrationTask

    original = IntegrationTask._IntegrationTask__saveAsPoni
    last_saved = None

    def save(task):
        filename = task.model().experimentSettingsModel().poniFile()

        def saved():
            nonlocal last_saved
            if not filename.isSynchronized():
                return
            try:
                path = Path(filename.value()).resolve()
                with path.open("rb") as stream:
                    data = stream.read((1 << 20) + 1)
                if len(data) > 1 << 20:
                    raise ValueError("saved PONI exceeds the size limit")
                last_saved = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
            except Exception as error:
                # Do not accidentally report an earlier save after a later
                # successful write whose contents could not be captured.
                last_saved = {"error": str(error)}

        filename.changed.connect(saved)
        try:
            return original(task)
        finally:
            filename.changed.disconnect(saved)

    IntegrationTask._IntegrationTask__saveAsPoni = save
    try:
        yield
    finally:
        IntegrationTask._IntegrationTask__saveAsPoni = original
        Path(report_path).write_text(json.dumps(last_saved), encoding="utf-8")


def _get_poni_with_explicit_parallax(self):
    from pyFAI.io.ponifile import PoniFile

    geometry = self._geometryTabs.geometryModel()
    settings = self.model().experimentSettingsModel()
    detector = settings.detector()
    values = {
        "dist": geometry.distance().value(),
        "poni1": geometry.poni1().value(),
        "poni2": geometry.poni2().value(),
        "rot1": geometry.rotation1().value(),
        "rot2": geometry.rotation2().value(),
        "rot3": geometry.rotation3().value(),
        "wavelength": geometry.wavelength().value(),
        "detector": detector.__class__.__name__,
        "detector_config": detector.get_config(),
    }
    if detector.sensor is not None:
        values["parallax"] = bool(settings.parallaxCorrection().value())
    return PoniFile(values)


def apply_calibration_backports() -> bool:
    """Patch only the pinned release, before the upstream GUI starts."""
    from xrd_tools.integrate._pyfai_rayonix import (
        apply_rayonix_serialization_backport,
    )

    if not apply_rayonix_serialization_backport():
        return False
    from pyFAI.gui.tasks.IntegrationTask import IntegrationTask

    IntegrationTask._IntegrationTask__getPoni = _get_poni_with_explicit_parallax
    return True


def main():
    apply_calibration_backports()
    from pyFAI.app.calib2 import main as upstream_main

    report_path = os.environ.get(CALIBRATION_SAVE_REPORT_ENV)
    if report_path:
        with capture_poni_saves(report_path):
            return upstream_main()
    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
