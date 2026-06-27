"""
Azimuthal integration, GID, calibration, and batch processing.
"""
from xrd_tools.integrate.batch import (
    DirectoryWatcher,
    process_scan,
    process_series,
)
from xrd_tools.integrate.gid import (
    create_fiber_integrator,
    integrate_gi_1d,
    integrate_gi_2d,
    integrate_gi_azimuthal_1d,
    integrate_gi_exitangles,
    integrate_gi_exitangles_1d,
    integrate_gi_polar,
    integrate_gi_polar_1d,
)
from xrd_tools.integrate.stitch_hist import (
    pyfai_gi_q_frames,
    pyfai_q_frames,
    stitch_q_grid,
    xu_q_frames,
)
from xrd_tools.integrate.multi import (
    create_multigeometry_integrators,
    create_multigeometry_integrators_from_geometry,
    stitch_1d,
    stitch_2d,
)
from xrd_tools.integrate.single import (
    integrate_1d,
    integrate_2d,
    integrate_radial,
    integrate_scan,
)
from xrd_tools.integrate.calibration import (
    detector_calibration_to_integrator,
    get_detector,
    get_detector_mask,
    load_poni,
    poni_to_fiber_integrator,
    poni_to_integrator,
    save_poni,
)
from xrd_tools.integrate.refine import (
    ControlFrame,
    RefineResult,
    refine_goniometer,
)

__all__ = [
    "DirectoryWatcher",
    "create_fiber_integrator",
    "create_multigeometry_integrators",
    "create_multigeometry_integrators_from_geometry",
    "pyfai_gi_q_frames",
    "pyfai_q_frames",
    "stitch_q_grid",
    "xu_q_frames",
    "integrate_gi_1d",
    "integrate_gi_2d",
    "integrate_gi_azimuthal_1d",
    "integrate_gi_exitangles",
    "integrate_gi_exitangles_1d",
    "integrate_gi_polar",
    "integrate_gi_polar_1d",
    "get_detector",
    "integrate_1d",
    "integrate_2d",
    "integrate_radial",
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
    "ControlFrame",
    "RefineResult",
    "refine_goniometer",
    "detector_calibration_to_integrator",
]
