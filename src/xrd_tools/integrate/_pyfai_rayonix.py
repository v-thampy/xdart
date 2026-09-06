"""Temporary pyFAI 2026.5.0 serialization backport; no integration changes.

Upstream PR https://github.com/silx-kit/pyFAI/pull/2905, merged as
6273a0d08be54005239ad55e0606147398d8cbfa. Remove this module and its callers
after adopting and validating an official release containing the fix.

The two replacement methods are derived from pyFAI (MIT), copyright European
Synchrotron Radiation Facility, Grenoble, France; see LICENSE-pyFAI-backport.
Imports remain lazy, including at the calibration admission boundary.
"""


def _rayonix_get_config(self):
    config = {
        "pixel1": self._pixel1,
        "pixel2": self._pixel2,
        "orientation": self.orientation or 3,
    }
    if getattr(self, "sensor", None) is not None:
        config["sensor"] = self.sensor.as_dict()
    return config


def _encoder_default(self, obj):
    from json import JSONEncoder

    import numpy
    from pyFAI import units
    # sensors.py uses this encoder for repr; keep this import deferred.
    from pyFAI.detectors.sensors import SensorConfig

    if isinstance(obj, units.Unit):
        return obj.name
    if isinstance(obj, numpy.generic):
        return obj.item()
    if isinstance(obj, SensorConfig):
        return obj.as_dict()
    return JSONEncoder.default(self, obj)


def apply_rayonix_serialization_backport() -> bool:
    """Apply the two upstream method fixes only to the affected pinned release.

    Repeated calls assign the same functions, never stack wrappers. No installed
    package files are edited, and later releases retain their own methods.
    """
    import pyFAI

    if pyFAI.version != "2026.5.0":
        return False
    from pyFAI.detectors._rayonix import _Rayonix
    from pyFAI.io._json import PyFAIEncoder

    _Rayonix.get_config = _rayonix_get_config
    PyFAIEncoder.default = _encoder_default
    return True
