#!/usr/bin/env python3
"""Validate Session-1 G18: S-4 standard-mode chi-offset written data.

This is the real-data ship gate for the S-4 chi-axis fixes.  It writes two
temporary processed NeXus scans and reads them back before comparing:

* Mode-A 1D chi output with an explicit partial panel chi range.
* q-axis 1D with the same explicit panel chi wedge, compared to the 2D cake.

By default the script looks for the private/local del-only LaB6 fixture under
``$XDART_TEST_DATA/stitching/data_del_only`` or ``~/repos/test_data``.  Override
``--image`` and ``--poni`` for the maintainer's canonical Session-1 data.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_IMAGE = "images/b_thampy_LaB6_16keV_scan7_0030.raw"
DEFAULT_PONI = "calibration/final_geometry.poni"


def _parse_pair(value: str) -> tuple[float, float]:
    try:
        left, right = value.replace(",", " ").split()
        return float(left), float(right)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            "expected two numbers, e.g. '0 90' or '0,90'"
        ) from exc


def _default_base() -> Path:
    roots: list[Path] = []
    env = os.environ.get("XDART_TEST_DATA")
    if env:
        roots.append(Path(env))
    roots.extend([
        Path.home() / "repos" / "test_data",
        Path.home() / "repos" / "example_notebooks",
    ])
    for root in roots:
        for base in (root / "stitching" / "data_del_only",
                     root / "Stitching" / "data_del_only"):
            if (base / DEFAULT_IMAGE).exists() and (base / DEFAULT_PONI).exists():
                return base
    return roots[0] / "stitching" / "data_del_only" if roots else Path(".")


def _build_parser() -> argparse.ArgumentParser:
    base = _default_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=base / DEFAULT_IMAGE,
                        help="raw detector image to validate")
    parser.add_argument("--poni", type=Path, default=base / DEFAULT_PONI,
                        help="calibration .poni for the image")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="directory for written validation .nxs files")
    parser.add_argument("--keep-output", action="store_true",
                        help="keep temporary .nxs files when --out-dir is not set")
    parser.add_argument("--shape", type=int, nargs=2, default=None,
                        metavar=("ROWS", "COLS"),
                        help="raw binary shape; defaults to the PONI detector shape")
    parser.add_argument("--raw-dtype", default="int32",
                        help="dtype for raw binary fallback reads")
    parser.add_argument("--threshold", type=float, default=8e5,
                        help="pixels above this value are masked")
    parser.add_argument("--chi-range", type=_parse_pair, default=(0.0, 90.0),
                        help="panel-frame explicit chi range, degrees")
    parser.add_argument("--q-range", type=_parse_pair, default=(1.0, 5.0),
                        help="radial q range, A^-1")
    parser.add_argument("--chi-offset", type=float, default=90.0,
                        help="standard-mode chi offset, degrees")
    parser.add_argument("--n-q", type=int, default=240,
                        help="q bins for the q-wedge check")
    parser.add_argument("--n-chi", type=int, default=180,
                        help="chi bins for the chi-output check")
    parser.add_argument("--reference-chi", type=float, default=None,
                        help="optional team-reference peak chi, degrees")
    parser.add_argument("--reference-tol", type=float, default=1.5,
                        help="absolute tolerance for --reference-chi, degrees")
    parser.add_argument("--min-correlation", type=float, default=0.94,
                        help="minimum profile correlation for 1D vs 2D comparisons")
    return parser


def _skip(message: str) -> int:
    print(f"SKIP: {message}")
    return 0


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _finite_pair(a, b)
    if x.size < 3:
        return float("nan")
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else float("nan")


def _peak(axis: np.ndarray, values: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(axis) & np.isfinite(values)
    if not finite.any():
        raise AssertionError("profile has no finite points")
    idx = int(np.nanargmax(np.where(finite, values, -np.inf)))
    return float(axis[idx])


def _bin_step(axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2:
        return 0.0
    return float(np.nanmedian(np.abs(np.diff(axis))))


def _nanmean_axis(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )


def _load_inputs(args):
    from xrd_tools.integrate.calibration import load_poni, poni_to_integrator
    from xrd_tools.io.image import read_image

    if not args.image.exists():
        return None, None, None, _skip(f"image not found: {args.image}")
    if not args.poni.exists():
        return None, None, None, _skip(f"PONI not found: {args.poni}")

    poni = load_poni(args.poni)
    ai = poni_to_integrator(poni)
    shape = tuple(args.shape) if args.shape else tuple(ai.detector.shape)
    image = read_image(
        args.image,
        detector_shape=shape,
        raw_dtype=args.raw_dtype,
        threshold=args.threshold,
    )
    return poni, ai, image, None


def _write_scan(path: Path, name: str, image: np.ndarray, ai, plan) -> None:
    from xrd_tools.reduction import Frame, NexusSink, Scan, run_reduction

    scan = Scan(name, [Frame(0, image=image)], integrator=ai)
    run_reduction(plan, scan, NexusSink(path, overwrite=True))


def _plan_chi(args):
    from xrd_tools.reduction import (
        Integration1DPlan,
        Integration2DPlan,
        ReductionPlan,
    )

    return ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=args.n_chi,
            unit="chi_deg",
            method="csr",
            radial_range=args.q_range,
            azimuth_range=args.chi_range,
            azimuth_offset=args.chi_offset,
            extra={"correctSolidAngle": False},
        ),
        integration_2d=Integration2DPlan(
            npt_rad=args.n_q,
            npt_azim=args.n_chi,
            unit="q_A^-1",
            method="csr",
            radial_range=args.q_range,
            azimuth_range=args.chi_range,
            azimuth_offset=args.chi_offset,
            extra={"correctSolidAngle": False},
        ),
    )


def _plan_q_wedge(args):
    from xrd_tools.reduction import (
        Integration1DPlan,
        Integration2DPlan,
        ReductionPlan,
    )

    return ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=args.n_q,
            unit="q_A^-1",
            method="csr",
            radial_range=args.q_range,
            azimuth_range=args.chi_range,
            azimuth_offset=args.chi_offset,
            extra={"correctSolidAngle": False},
        ),
        integration_2d=Integration2DPlan(
            npt_rad=args.n_q,
            npt_azim=1,
            unit="q_A^-1",
            method="csr",
            radial_range=args.q_range,
            azimuth_range=args.chi_range,
            azimuth_offset=args.chi_offset,
            extra={"correctSolidAngle": False},
        ),
    )


def _check_written_outputs(args, chi_path: Path, q_path: Path) -> bool:
    from xrd_tools.io import get_1d, get_2d

    checks: list[tuple[bool, str]] = []

    chi_1d = get_1d(chi_path, frame=0)
    chi_2d = get_2d(chi_path, frame=0)
    checks.append((
        chi_1d.q_unit == "chi_deg",
        f"1D chi unit is {chi_1d.q_unit!r}",
    ))
    checks.append((
        np.allclose(chi_1d.q, chi_2d.chi, rtol=1e-6, atol=1e-6),
        "written 1D chi axis matches written 2D cake chi axis",
    ))
    # Convenience readers expose 2D cakes as (chi, q).  Collapse q to compare
    # the written chi profile against the simultaneously written cake.
    chi_from_cake = _nanmean_axis(chi_2d.intensity, axis=1)
    chi_corr = _correlation(chi_1d.intensity, chi_from_cake)
    chi_peak_delta = abs(_peak(chi_1d.q, chi_1d.intensity)
                         - _peak(chi_2d.chi, chi_from_cake))
    checks.append((
        np.isfinite(chi_corr) and chi_corr >= args.min_correlation,
        f"chi 1D-vs-2D profile correlation {chi_corr:.4f}",
    ))
    checks.append((
        chi_peak_delta <= max(2.0 * _bin_step(chi_1d.q), 1.0),
        f"chi peak delta {chi_peak_delta:.3f} deg",
    ))
    if args.reference_chi is not None:
        chi_peak = _peak(chi_1d.q, chi_1d.intensity)
        ref_delta = abs(chi_peak - args.reference_chi)
        checks.append((
            ref_delta <= args.reference_tol,
            f"team-reference chi peak delta {ref_delta:.3f} deg",
        ))

    q_1d = get_1d(q_path, frame=0)
    q_2d = get_2d(q_path, frame=0)
    q_profile = q_2d.intensity[0, :]
    q_corr = _correlation(q_1d.intensity, q_profile)
    q_peak_delta = abs(_peak(q_1d.q, q_1d.intensity)
                       - _peak(q_2d.q, q_profile))
    checks.append((
        q_1d.q_unit == q_2d.q_unit == "q_A^-1",
        f"q units are 1D={q_1d.q_unit!r}, 2D={q_2d.q_unit!r}",
    ))
    checks.append((
        np.allclose(q_1d.q, q_2d.q, rtol=1e-6, atol=1e-8),
        "q-wedge 1D radial axis matches written 2D cake q axis",
    ))
    checks.append((
        np.allclose(q_2d.chi, [sum(args.chi_range) / 2.0],
                    rtol=1e-6, atol=max(_bin_step(q_2d.chi), 1e-6)),
        "q-wedge 2D cake uses the requested panel chi wedge",
    ))
    checks.append((
        np.isfinite(q_corr) and q_corr >= args.min_correlation,
        f"q-wedge 1D-vs-2D profile correlation {q_corr:.4f}",
    ))
    checks.append((
        q_peak_delta <= max(2.0 * _bin_step(q_1d.q), 0.03),
        f"q-wedge peak delta {q_peak_delta:.4f} A^-1",
    ))

    ok = True
    print("\n=== G18 S-4 written-data validation ===")
    for passed, message in checks:
        ok = ok and bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {message}")
    return ok


def _run_with_output_dir(args, out_dir: Path) -> int:
    poni, ai, image, skipped = _load_inputs(args)
    if skipped is not None:
        return skipped

    out_dir.mkdir(parents=True, exist_ok=True)
    chi_path = out_dir / "g18_s4_chi_axis.nxs"
    q_path = out_dir / "g18_s4_q_wedge.nxs"

    _write_scan(chi_path, "g18-s4-chi", image, ai, _plan_chi(args))
    _write_scan(q_path, "g18-s4-q-wedge", image, ai, _plan_q_wedge(args))
    ok = _check_written_outputs(args, chi_path, q_path)

    print(f"\nimage: {args.image}")
    print(f"poni:  {args.poni}")
    print(f"written: {chi_path}")
    print(f"written: {q_path}")
    print(f"\nG18 S-4 validation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.out_dir is not None:
        return _run_with_output_dir(args, args.out_dir)

    tmp = tempfile.TemporaryDirectory(prefix="g18_s4_chi_")
    try:
        code = _run_with_output_dir(args, Path(tmp.name))
        if args.keep_output:
            kept = Path.cwd() / Path(tmp.name).name
            if kept.exists():
                raise SystemExit(f"refusing to overwrite {kept}")
            Path(tmp.name).rename(kept)
            print(f"kept output under {kept}")
            tmp = None  # type: ignore[assignment]
        return code
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
