"""
Azimuthal integration, GID, calibration, and batch processing.
"""
from ssrl_xrd_tools.integrate.batch import (
    DirectoryWatcher,
    process_scan,
    process_series,
)
from ssrl_xrd_tools.integrate.gid import (
    create_fiber_integrator,
    integrate_gi_1d,
    integrate_gi_2d,
    integrate_gi_exitangles,
    integrate_gi_exitangles_1d,
    integrate_gi_polar,
    integrate_gi_polar_1d,
)
from ssrl_xrd_tools.integrate.multi import (
    create_multigeometry_integrators,
    stitch_1d,
    stitch_2d,
)
from ssrl_xrd_tools.integrate.single import (
    integrate_1d,
    integrate_2d,
    integrate_scan,
)
from ssrl_xrd_tools.integrate.calibration import (
    get_detector,
    get_detector_mask,
    load_poni,
    poni_to_fiber_integrator,
    poni_to_integrator,
    save_poni,
)

__all__ = [
    "DirectoryWatcher",
    "create_fiber_integrator",
    "create_multigeometry_integrators",
    "integrate_gi_1d",
    "integrate_gi_2d",
    "integrate_gi_exitangles",
    "integrate_gi_exitangles_1d",
    "integrate_gi_polar",
    "integrate_gi_polar_1d",
    "get_detector",
    "integrate_1d",
    "integrate_2d",
    "integrate_scan",
    "stitch_1d",
    "stitch_2d",
    "get_detector_mask",
    "load_poni",
    "poni_to_fiber_integrator",
    "poni_to_integrator",
    "process_scan",
    "process_series",
    "save_poni",
]
