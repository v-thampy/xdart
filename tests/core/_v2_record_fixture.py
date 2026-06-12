"""Deterministic scan -> .nxs via the (xdart) v2 writer, for the 6a gate.

Used both to CAPTURE the pre-6a reference signature and by the gate test
to re-write the identical scan post-refactor.
"""
from __future__ import annotations

import os
import numpy as np


def write_reference_scan(out_path, source_base):
    """Write a small deterministic scan exercising the complete v2 record:
    integrated 1D+2D stacks, per-frame thumbnails + source refs (relative
    to ``source_base``), instrument, provenance, mask, scan_data."""
    import pandas as pd
    from tests.xdart.test_nexus_writer_roundtrip import _DuckArch, _DuckSphere
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    os.makedirs(source_base, exist_ok=True)
    frames = []
    for i in range(3):
        fr = _DuckArch(idx=i, seed=7)
        # Deterministic thumbnail + a source file INSIDE the project root
        rng = np.random.default_rng(100 + i)
        fr.thumbnail = rng.random((32, 30)).astype(np.float32)
        src = os.path.join(source_base, f"frame_{i:04d}.tif")
        with open(src, "wb") as fh:
            fh.write(b"not-a-real-tiff")          # pointer target only
        fr.source_file = src
        fr.source_frame_idx = 0
        frames.append(fr)

    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 12.0, 3).astype(np.float32),
        "i0": np.linspace(1e6, 1.1e6, 3).astype(np.float32),
    })
    scan = _DuckSphere(frames, scan_data=scan_data,
                       global_mask=np.array([0, 7, 64], dtype=np.int64))
    scan.source_base = source_base
    save_scan_to_nexus(scan, out_path, mode="w", finalize=True)
    return out_path
