"""SPEC file parsing: scan metadata, energy, UB, angles."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from silx.io.specfile import SpecFile

logger = logging.getLogger(__name__)


def _get_spec_scan(spec_file: Path | str, scan_num: str):
    return SpecFile(str(spec_file))[scan_num]


def get_scan_path_info(scan: str) -> tuple[str, str]:
    """
    Parse scan name into sample name and SPEC-compatible scan number.
    """
    idx = scan.rfind("_")
    if idx < 0:
        return scan, "1.1"
    sample_name = scan[:idx]
    scan_num = scan[idx + 1 :]
    return sample_name, f"{scan_num}.1"


def get_energy_and_UB(spec_file: Path | str, scan_num: str) -> tuple[float, np.ndarray]:
    scan_data = _get_spec_scan(spec_file, scan_num)
    energy = scan_data.motor_position_by_name("energy")
    header_dict = scan_data.scan_header_dict
    UB = np.array(header_dict["G3"].split(), dtype=float).reshape(3, 3)
    return energy, UB


def get_spec_scan_type(spec_file: Path | str, scan_num: str) -> str | list[str]:
    scan_data = _get_spec_scan(spec_file, scan_num)
    scan_hdr = scan_data.scan_header_dict["S"].split()
    x_col = scan_hdr[1]

    if x_col == "hklscan":
        hkl_ranges = np.array(
            [
                float(scan_hdr[3]) - float(scan_hdr[2]),
                float(scan_hdr[5]) - float(scan_hdr[4]),
                float(scan_hdr[7]) - float(scan_hdr[6]),
            ]
        )
        x_col = [x for x, r in zip(("H", "K", "L"), hkl_ranges != 0) if r]

    return x_col


def get_from_spec_file(
    fname: Path | str,
    scan_number: str,
    cols: list[str],
) -> dict[str, np.ndarray]:
    scan_data = _get_spec_scan(fname, scan_number)
    spec_data: dict[str, np.ndarray] = {}

    npts = scan_data.data.shape[1]

    for col in cols:
        if col in scan_data.labels:
            spec_data[col] = np.asarray(scan_data.data_column_by_name(col))
        elif col in scan_data.motor_names:
            spec_data[col] = np.full(npts, scan_data.motor_position_by_name(col))
        else:
            logger.warning(
                "Column/motor %s does not exist in scan %s",
                col,
                scan_number,
            )

    return spec_data


def get_angles(
    spec_file: Path | str,
    scan_num: str,
    diff_motors: list[str] | tuple[str, ...],
) -> list[list[float]]:
    scan_data = _get_spec_scan(spec_file, scan_num)
    npts = scan_data.data.shape[1]
    angles: list[list[float]] = []

    for motor in diff_motors:
        if motor in scan_data.labels:
            vals = scan_data.data_column_by_name(motor)
        else:
            vals = np.full(npts, scan_data.motor_position_by_name(motor))
        angles.append(list(vals))

    return angles
