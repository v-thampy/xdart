"""Standalone pyFAI calibration with temporary 2026.5.0 save corrections.

The PONI builder below is derived from pyFAI's IntegrationTask (MIT, ESRF;
see LICENSE-pyFAI-backport). Its only change is to preserve an explicit False
when a sensor is selected with parallax disabled. Remove with the qualified
upstream replacement; this is not a second calibration implementation.
"""


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

    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
