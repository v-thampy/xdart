"""Peak finding: extract_peaks, peak_table."""

from __future__ import annotations

from typing import Any

from scipy.signal import find_peaks
from ssrl_xrd_tools.core.containers import IntegrationResult1D


def extract_peaks(
    result: IntegrationResult1D,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
    width: float | None = None,
    **kwargs: Any
) -> dict[str, Any]:
    """
    Find peaks in 1D integration data using scipy.signal.find_peaks.
    
    Parameters
    ----------
    result : IntegrationResult1D
        The 1D XRD pattern.
    height : float, optional
        Minimum intensity height.
    distance : int, optional
        Minimum index distance between peaks.
    prominence : float, optional
        Minimum peak prominence.
    width : float, optional
        Minimum width of peaks.
    **kwargs
        Additional arguments passed to scipy.signal.find_peaks.

    Returns
    -------
    dict
        Dictionary containing:
        - 'indices': array of indices of the peaks
        - 'radial': array of radial axis values (e.g. Q or 2theta)
        - 'intensity': array of peak intensities
        - 'properties': properties dict returned by scipy
    """
    indices, properties = find_peaks(
        result.intensity,
        height=height,
        distance=distance,
        prominence=prominence,
        width=width,
        **kwargs
    )
    
    radial_vals = result.radial[indices]
    intensity_vals = result.intensity[indices]
    
    return {
        "indices": indices,
        "radial": radial_vals,
        "intensity": intensity_vals,
        "properties": properties,
    }


def peak_table(peaks_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert a peaks dictionary into a list of row dictionaries.

    Parameters
    ----------
    peaks_dict : dict
        Output from `extract_peaks`.

    Returns
    -------
    list of dict
        List of row dictionaries, each representing a single peak
        with its index, radial position, intensity, and any extracted
        properties (like width, prominence).
    """
    rows = []
    for i, idx in enumerate(peaks_dict["indices"]):
        row = {
            "index": idx,
            "radial": peaks_dict["radial"][i],
            "intensity": peaks_dict["intensity"][i],
        }
        for prop_name, prop_vals in peaks_dict["properties"].items():
            if len(prop_vals) == len(peaks_dict["indices"]):
                row[prop_name] = prop_vals[i]
        rows.append(row)
    return rows


__all__ = ["extract_peaks", "peak_table"]
