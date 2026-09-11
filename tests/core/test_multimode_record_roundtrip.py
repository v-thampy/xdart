"""Step 1 acceptance: multi-result per-mode NeXus persistence (ADR-0003/0005).

Pure-io (no xdart): write FrameRecords carrying >=2 GI modes per dimension
through ``write_frame_records``, reload via ``read_frame_records``, and assert
each ``(frame, mode)`` is byte-equivalent — the multi-mode reload-equivalence
gate.  Also pins: the nested-subgroup NXdata contract, the reader mode-selection
rule, the per-scan mode attrs, the byte-compat collapse for standard scans, the
current attr-free Standard layout, and the ``FrameView -> IntegrationResult``
transpose round-trip.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.core import (
    DEFAULT_MODE_KEY,
    Axis,
    FrameRecord,
    FrameView,
    TwoDKind,
    assert_framerecord_equivalent,
    assert_frameview_equivalent,
    axis_from_unit,
    view_to_result_1d,
    view_to_result_2d,
)
from xrd_tools.io import (
    FrameViewReader,
    read_frame_record,
    read_frame_records,
    read_frame_view,
    write_frame_records,
    write_integrated_stack,
)
from xrd_tools.io.schema import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
)


def _view(label, scale, *, nq=5, nchi=3, x1="q_A^-1", x2="qip_A^-1",
          y2="qoop_A^-1", kind=TwoDKind.QIP_QOOP):
    """A combined FrameView whose axis labels DERIVE from units (as real code
    does), so a reload — which re-derives labels from stored units — matches."""
    return FrameView(
        label=label,
        axis_1d=axis_from_unit(x1, np.linspace(1.0, 2.0, nq)),
        intensity_1d=np.arange(nq, dtype=float) * scale + label,
        axis_2d_x=axis_from_unit(x2, np.linspace(0.0, 1.0, nq)),
        axis_2d_y=axis_from_unit(y2, np.linspace(0.0, 1.0, nchi)),
        intensity_2d=np.arange(nchi * nq, dtype=float).reshape(nchi, nq) * scale + label,
        two_d_kind=kind,
        metadata_raw={"motor": float(label)},
    )


def _multimode_records(n=3):
    """n frames with 1D modes {q_total(primary), q_oop} and 2D modes
    {qip_qoop(primary), q_chi}.  q_oop has a DIFFERENT bin count (7 vs 5) and
    q_chi a different chi count (4 vs 3) so a wrong-subgroup read fails loud."""
    recs = []
    for fi in range(n):
        r = FrameRecord.from_view(_view(fi, 1.0), mode_1d="q_total", mode_2d="qip_qoop")
        r = r.with_result_1d("q_oop", _view(fi, 9.0, nq=7, x1="qoop_A^-1"),
                             make_active=False)
        r = r.with_result_2d(
            "q_chi",
            _view(fi, 5.0, nchi=4, x2="q_A^-1", y2="chi_deg", kind=TwoDKind.Q_CHI),
            make_active=False,
        )
        recs.append(r)
    return recs


def _write(entry_path, records):
    with h5py.File(entry_path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        write_frame_records(entry, records)


# --------------------------------------------------------------------------- #
# the multi-mode reload-equivalence gate (ADR-0005 Phase-5 criterion)
# --------------------------------------------------------------------------- #

def test_multimode_reload_equivalence(tmp_path):
    records = _multimode_records(3)
    p = str(tmp_path / "mm.nexus")
    _write(p, records)

    reloaded = read_frame_records(p)
    assert len(reloaded) == 3
    for orig, rl in zip(records, reloaded):
        assert_framerecord_equivalent(orig, rl)  # per (frame, mode) byte-equivalent


def test_partial_extra_mode_rows_roundtrip(tmp_path):
    records = []
    for fi in range(3):
        rec = FrameRecord.from_view(
            _view(fi, 1.0), mode_1d="q_total", mode_2d="qip_qoop",
        )
        if fi != 1:
            rec = rec.with_result_1d(
                "q_ip",
                _view(fi, 4.0, x1="qip_A^-1"),
                make_active=False,
            )
        records.append(rec)
    p = str(tmp_path / "partial_extra.nexus")
    _write(p, records)

    reloaded = read_frame_records(p)
    assert set(reloaded[0].modes_1d) == {"q_total", "q_ip"}
    assert set(reloaded[1].modes_1d) == {"q_total"}
    assert set(reloaded[2].modes_1d) == {"q_total", "q_ip"}
    np.testing.assert_allclose(
        reloaded[2].view_1d("q_ip").intensity_1d,
        records[2].view_1d("q_ip").intensity_1d,
    )


def test_nested_subgroup_nxdata_contract(tmp_path):
    p = str(tmp_path / "mm.nexus")
    _write(p, _multimode_records(2))
    with h5py.File(p, "r") as f:
        assert f["entry"].attrs[SCHEMA_NAME_ATTR] == "xrd_tools.processed_scan"
        assert f["entry"].attrs[SCHEMA_VERSION_ATTR] == 3
        for grp_path, axes in (
            ("entry/integrated_1d", ("frame_index", "axis_1")),
            ("entry/integrated_1d/q_oop", ("frame_index", "axis_1")),
            ("entry/integrated_2d", ("frame_index", "axis_2", "axis_1")),
            ("entry/integrated_2d/q_chi", ("frame_index", "axis_2", "axis_1")),
        ):
            g = f[grp_path]
            assert g.attrs["NX_class"] == "NXdata", grp_path
            assert g.attrs["signal"] == "intensity", grp_path
            assert tuple(g.attrs["axes"]) == axes, grp_path
            assert {"intensity", *axes}.issubset(g), grp_path
            assert "q" not in g and "chi" not in g, grp_path
        # distinct on-disk shapes per mode (the wrong-subgroup tripwire)
        assert f["entry/integrated_1d/intensity"].shape[1] == 5
        assert f["entry/integrated_1d/q_oop/intensity"].shape[1] == 7
        assert f["entry/integrated_2d/intensity"].shape[1:] == (3, 5)
        assert f["entry/integrated_2d/q_chi/intensity"].shape[1:] == (4, 5)


def test_frozen_validators_accept_each_group_standalone(tmp_path):
    """Each nested mode list passes the FROZEN uniform-axes validators on its
    own — the validators are reused per group, never loosened."""
    from xrd_tools.io.nexus import _require_uniform_axes_1d, _require_uniform_axes_2d

    recs = _multimode_records(2)
    r1 = [view_to_result_1d(r.results_1d["q_oop"]) for r in recs]
    r2 = [view_to_result_2d(r.results_2d["q_chi"]) for r in recs]
    _require_uniform_axes_1d(r1)   # must not raise
    _require_uniform_axes_2d(r2)


def test_reader_mode_selection_rule(tmp_path):
    p = str(tmp_path / "mm.nexus")
    _write(p, _multimode_records(2))
    with FrameViewReader(p) as rd:
        assert rd.is_multi_mode() is True
        assert rd.modes_1d() == ("q_total", "q_oop")      # primary first
        assert rd.modes_2d() == ("qip_qoop", "q_chi")
        assert rd.primary_mode_1d() == "q_total"
        assert rd.primary_mode_2d() == "qip_qoop"
        # mode == primary -> top-level; else -> nested subgroup (distinct data)
        assert rd.read(0, mode_1d="q_total").intensity_1d.shape[0] == 5
        assert rd.read(0, mode_1d="q_oop").intensity_1d.shape[0] == 7
        # default (no mode arg) == primary
        assert rd.read(0).intensity_1d.shape[0] == 5
        # unknown mode -> that dimension absent, no crash
        assert not rd.read(0, mode_1d="bogus").has_1d
        assert not rd.read(0, mode_2d="bogus").has_2d


def test_mode_attrs_present_and_ordered(tmp_path):
    p = str(tmp_path / "mm.nexus")
    _write(p, _multimode_records(2))
    with h5py.File(p, "r") as f:
        g1 = f["entry/integrated_1d"]
        g2 = f["entry/integrated_2d"]
        assert g1.attrs["primary_mode"] == "q_total"
        assert list(g1.attrs["multi_result_modes"]) == ["q_total", "q_oop"]
        assert g2.attrs["primary_mode"] == "qip_qoop"
        assert list(g2.attrs["multi_result_modes"]) == ["qip_qoop", "q_chi"]


def test_transpose_roundtrip():
    """from_results(view_to_result_*(v)) reconstructs an equivalent view (the
    (ny,nx)<->(radial,azimuthal) transpose is its own inverse).  Both sides go
    through from_results so derived metadata_numeric matches."""
    src = _view(0, 2.0)
    v = FrameView.from_results(
        label=0, result_1d=view_to_result_1d(src), result_2d=view_to_result_2d(src),
        metadata_raw={"motor": 0.0},
    )
    rebuilt = FrameView.from_results(
        label=0, result_1d=view_to_result_1d(v), result_2d=view_to_result_2d(v),
        metadata_raw={"motor": 0.0},
    )
    assert_frameview_equivalent(v, rebuilt)


# --------------------------------------------------------------------------- #
# current Standard / single-mode collapse
# --------------------------------------------------------------------------- #

def test_standard_record_collapse_is_byte_identical(tmp_path):
    """A standard (DEFAULT-mode) record stack written via write_frame_records is
    byte-identical to the same data via plain write_integrated_stack, and stamps
    NO new attr / subgroup."""
    from tests.core.h5sig import h5_content_signature

    std = [FrameRecord.from_view(_view(fi, 1.0)) for fi in range(2)]  # default mode
    pa = str(tmp_path / "a.nexus")
    pb = str(tmp_path / "b.nexus")
    _write(pa, std)
    with h5py.File(pb, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        write_integrated_stack(
            entry, frame_indices=[0, 1],
            results_1d=[view_to_result_1d(_view(fi, 1.0)) for fi in range(2)],
            results_2d=[view_to_result_2d(_view(fi, 1.0)) for fi in range(2)],
        )
    assert h5_content_signature(pa) == h5_content_signature(pb)
    with h5py.File(pa, "r") as f:
        assert "primary_mode" not in f["entry/integrated_1d"].attrs
        assert "multi_result_modes" not in f["entry/integrated_1d"].attrs
        # no nested children
        assert not any(isinstance(v, h5py.Group)
                       for v in f["entry/integrated_1d"].values())


def test_current_standard_single_mode_uses_default_layout(tmp_path):
    """Current attr-free Standard output reads as one DEFAULT-mode record."""
    p = str(tmp_path / "standard.nexus")
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        write_integrated_stack(
            entry, frame_indices=[0, 1],
            results_1d=[view_to_result_1d(_view(fi, 1.0)) for fi in range(2)],
            results_2d=[view_to_result_2d(_view(fi, 1.0)) for fi in range(2)],
        )
    with FrameViewReader(p) as rd:
        assert rd.is_multi_mode() is False
        assert rd.primary_mode_1d() == DEFAULT_MODE_KEY
        assert rd.modes_1d() == (DEFAULT_MODE_KEY,)
        assert not rd.read(0, mode_1d="q_oop").has_1d   # graceful
    rec = read_frame_record(p, 0)
    assert_framerecord_equivalent(rec, FrameRecord.from_view(read_frame_view(p, 0)))


# --------------------------------------------------------------------------- #
# fail-loud guards on the public write_integrated_stack (adversarial review P2)
# --------------------------------------------------------------------------- #

def test_extra_mode_colliding_with_primary_is_rejected(tmp_path):
    """P2-A: a primary_mode that also appears in extra_modes would write the
    same key at two locations (top-level + subgroup) → silent loss; reject."""
    r1 = [view_to_result_1d(_view(fi, 1.0)) for fi in range(2)]
    r1b = [view_to_result_1d(_view(fi, 9.0, nq=7, x1="qoop_A^-1")) for fi in range(2)]
    p = str(tmp_path / "x.nexus")
    with h5py.File(p, "w") as f:
        with pytest.raises(ValueError, match="must not also appear"):
            write_integrated_stack(
                f.create_group("entry"), frame_indices=[0, 1],
                results_1d=r1, extra_modes_1d={"q_total": r1b},
                primary_mode_1d="q_total",
            )


def test_extras_require_named_primary(tmp_path):
    """P2-B: nested subgroups without a named primary would leave the file
    without its capability marker (is_multi_mode would lie); reject."""
    r1 = [view_to_result_1d(_view(fi, 1.0)) for fi in range(2)]
    r1b = [view_to_result_1d(_view(fi, 9.0, nq=7, x1="qoop_A^-1")) for fi in range(2)]
    p = str(tmp_path / "x.nexus")
    with h5py.File(p, "w") as f:
        with pytest.raises(ValueError, match="named primary_mode_1d"):
            write_integrated_stack(
                f.create_group("entry"), frame_indices=[0, 1],
                results_1d=r1, extra_modes_1d={"q_oop": r1b},
                primary_mode_1d=None,  # would write q_oop/ but stamp no marker
            )


def test_partial_mode_set_rejected_with_clear_error(tmp_path):
    """P3-E: a per-frame-varying mode set (one frame has 1D, another doesn't)
    is rejected with a precise message, not a downstream length mismatch."""
    full = FrameRecord.from_view(_view(0, 1.0), mode_1d="q_total", mode_2d="qip_qoop")
    only_2d = FrameRecord(
        label=1, results_2d=dict(full.results_2d), active_mode_2d="qip_qoop",
    )  # frame 1 has no 1D results while frame 0 does
    p = str(tmp_path / "x.nexus")
    with h5py.File(p, "w") as f:
        with pytest.raises(ValueError, match="no 1D results"):
            write_frame_records(f.create_group("entry"), [full, only_2d])


def test_reader_rejects_unreadable_declared_child_mode(tmp_path):
    """A malformed declared result invalidates the complete current artifact."""
    p = str(tmp_path / "mm.nexus")
    _write(p, _multimode_records(2))
    # hand-corrupt: drop the q_oop child's intensity dataset
    with h5py.File(p, "r+") as f:
        del f["entry/integrated_1d/q_oop/intensity"]
    with pytest.raises(ValueError, match="not a current xdart"):
        with FrameViewReader(p):
            pass


def test_single_named_gi_mode_persists_active_mode(tmp_path):
    """A single NAMED GI mode (no extras) still records primary_mode so the
    active mode round-trips (additive on GI files; non-GI stays byte-clean)."""
    recs = [FrameRecord.from_view(_view(fi, 1.0), mode_1d="q_total",
                                  mode_2d="qip_qoop") for fi in range(2)]
    p = str(tmp_path / "single_gi.nexus")
    _write(p, recs)
    with h5py.File(p, "r") as f:
        assert f["entry/integrated_1d"].attrs["primary_mode"] == "q_total"
        assert not any(isinstance(v, h5py.Group)
                       for v in f["entry/integrated_1d"].values())  # no subgroups
    for orig, rl in zip(recs, read_frame_records(p)):
        assert_framerecord_equivalent(orig, rl)
