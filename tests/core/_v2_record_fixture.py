"""Headless deterministic records for the frozen v2 compatibility gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io import NexusRecordWriter, RecordWrite, WriterFinalization
from xrd_tools.io.nexus import resolve_stack_compression


N_FRAMES = 3
N_Q = 32
N_CHI = 16


def _result_1d(index: int, *, unit: str = "q_A^-1", offset: int = 0):
    rng = np.random.default_rng(7 + index + 1000 * offset)
    radial = np.linspace(0.5, 5.0, N_Q, dtype=np.float32)
    return IntegrationResult1D(
        radial=radial,
        intensity=rng.random(N_Q, dtype=np.float32),
        sigma=rng.random(N_Q, dtype=np.float32) * np.float32(0.1),
        unit=unit,
    )


def _result_2d(
    index: int,
    *,
    unit: str = "q_A^-1",
    azimuthal_unit: str = "deg",
    offset: int = 0,
):
    # Preserve the legacy standard fixture's RNG sequence exactly: its 2-D
    # values followed the 1-D intensity and sigma draws from the same stream.
    rng = np.random.default_rng(7 + index + 1000 * offset)
    rng.random(N_Q, dtype=np.float32)
    rng.random(N_Q, dtype=np.float32)
    radial = np.linspace(0.5, 5.0, N_Q, dtype=np.float32)
    azimuthal = np.linspace(
        -180.0, 180.0, N_CHI, endpoint=False, dtype=np.float32
    )
    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=rng.random((N_Q, N_CHI), dtype=np.float32),
        unit=unit,
        azimuthal_unit=azimuthal_unit,
    )


def _thumbnail(index: int) -> np.ndarray:
    return np.random.default_rng(100 + index).random((32, 30)).astype(np.float32)


def _scan_data():
    import pandas as pd

    return pd.DataFrame(
        {
            "tth": np.linspace(10.0, 12.0, N_FRAMES).astype(np.float32),
            "i0": np.linspace(1e6, 1.1e6, N_FRAMES).astype(np.float32),
        },
        index=range(N_FRAMES),
    )


def _source_rows(source_base: Path) -> tuple[RecordWrite, ...]:
    scan_data = _scan_data()
    rows = []
    for index in range(N_FRAMES):
        source = source_base / f"frame_{index:04d}.tif"
        source.write_bytes(b"not-a-real-tiff")
        rows.append(
            RecordWrite(
                label=index,
                result_1d=_result_1d(index),
                result_2d=_result_2d(index),
                thumbnail=_thumbnail(index),
                source_path=source,
                source_frame_index=0,
                metadata={
                    name: scan_data.loc[index, name] for name in scan_data.columns
                },
            )
        )
    return tuple(rows)


def _standard_finalization() -> WriterFinalization:
    return WriterFinalization(
        scan_data=_scan_data(),
        frame_indices=tuple(range(N_FRAMES)),
        provenance_config={
            "bai_1d_args": {"numpoints": N_Q},
            "bai_2d_args": {"npt_rad": N_Q, "npt_azim": N_CHI},
            "gi": False,
        },
        provenance_inputs={},
        program="xdart",
        detector_calibration={
            "dist": 0.1,
            "poni1": 0.05,
            "poni2": 0.05,
            "rot1": 0.0,
            "rot2": 0.0,
            "rot3": 0.0,
        },
        global_mask=np.array([0, 7, 64], dtype=np.int64),
    )


def write_reference_scan(out_path, source_base, *, compression="default"):
    """Write the exact pre-6a Standard record through the shared writer."""
    source_base = Path(source_base)
    source_base.mkdir(parents=True, exist_ok=True)
    codec = (
        resolve_stack_compression()
        if compression == "default"
        else compression
    )
    writer = NexusRecordWriter(
        out_path,
        compression=codec,
        overwrite=True,
        flush_every=None,
        source_base=source_base,
    )
    writer.begin()
    writer.write_batch(_source_rows(source_base))
    writer.finish(_standard_finalization())
    return out_path


_GI_1D_UNITS = {
    "q_total": "qtot_A^-1",
    "q_ip": "qip_A^-1",
    "q_oop": "qoop_A^-1",
    "exit_angle": "exit_angle_deg",
    "chi_gi": "chigi_deg",
}
_GI_2D_UNITS = {
    "qip_qoop": ("qip_A^-1", "qoop_A^-1"),
    "q_chi": ("qtot_A^-1", "chigi_deg"),
    "exit_angles": ("exit_angle_horz_deg", "exit_angle_vert_deg"),
}


def write_gi_reference_scan(out_path, source_base, *, compression="default"):
    """Write a local multi-mode GI record through the same shared writer."""
    source_base = Path(source_base)
    source_base.mkdir(parents=True, exist_ok=True)
    codec = (
        resolve_stack_compression()
        if compression == "default"
        else compression
    )
    writer = NexusRecordWriter(
        out_path,
        compression=codec,
        overwrite=True,
        flush_every=None,
        source_base=source_base,
    )
    writer.begin(primary_mode_1d="q_total", primary_mode_2d="qip_qoop")
    base = _source_rows(source_base)
    writer.write_batch(
        tuple(
            RecordWrite(
                label=row.label,
                result_1d=_result_1d(
                    row.label, unit=_GI_1D_UNITS["q_total"], offset=1
                ),
                result_2d=_result_2d(
                    row.label,
                    unit=_GI_2D_UNITS["qip_qoop"][0],
                    azimuthal_unit=_GI_2D_UNITS["qip_qoop"][1],
                    offset=1,
                ),
                mode_1d="q_total",
                mode_2d="qip_qoop",
                source_path=row.source_path,
                source_frame_index=row.source_frame_index,
                thumbnail=row.thumbnail,
                metadata=row.metadata,
            )
            for row in base
        )
    )
    for offset, mode in enumerate(
        ("q_ip", "q_oop", "exit_angle", "chi_gi"), start=2
    ):
        writer.write_batch(
            tuple(
                RecordWrite(
                    label=index,
                    result_1d=_result_1d(
                        index, unit=_GI_1D_UNITS[mode], offset=offset
                    ),
                    mode_1d=mode,
                    source_path=base[index].source_path,
                    source_frame_index=base[index].source_frame_index,
                    write_frame_record=False,
                )
                for index in range(N_FRAMES)
            )
        )
    for offset, mode in enumerate(("q_chi", "exit_angles"), start=6):
        writer.write_batch(
            tuple(
                RecordWrite(
                    label=index,
                    result_2d=_result_2d(
                        index,
                        unit=_GI_2D_UNITS[mode][0],
                        azimuthal_unit=_GI_2D_UNITS[mode][1],
                        offset=offset,
                    ),
                    mode_2d=mode,
                    source_path=base[index].source_path,
                    source_frame_index=base[index].source_frame_index,
                    write_frame_record=False,
                )
                for index in range(N_FRAMES)
            )
        )
    writer.finish(
        WriterFinalization(
            scan_data=_scan_data(),
            frame_indices=tuple(range(N_FRAMES)),
            provenance_config={
                "bai_1d_args": {"numpoints": N_Q, "gi_mode_1d": "q_total"},
                "bai_2d_args": {
                    "npt_rad": N_Q,
                    "npt_azim": N_CHI,
                    "gi_mode_2d": "qip_qoop",
                },
                "gi": True,
                "gi_config": {
                    "gi_mode_1d": "q_total",
                    "gi_mode_2d": "qip_qoop",
                    "incidence_motor": "th",
                    "th_val": 0.2,
                    "tilt_angle": 0.0,
                    "sample_orientation": 1,
                },
            },
            provenance_inputs={},
            program="xdart",
            detector_calibration={
                "dist": 0.1,
                "poni1": 0.05,
                "poni2": 0.05,
                "rot1": 0.0,
                "rot2": 0.0,
                "rot3": 0.0,
            },
            global_mask=np.array([0, 7, 64], dtype=np.int64),
        )
    )
    return out_path
