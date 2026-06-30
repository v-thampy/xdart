"""6a gate: the refactored writer produces a content-identical v2 record.

The committed fixture is the content signature of a deterministic scan
written by the PRE-6a (all-xdart) writer.  Any 6a step that changes what
lands on disk for an identical scan fails here.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("xdart", reason="gate exercises the GUI-side writer")

FIXTURE = Path(__file__).parent / "fixtures" / "v2_record_signature_pre6a.json"


def test_v2_record_content_identical_to_pre6a(tmp_path):
    from tests.core._v2_record_fixture import write_reference_scan
    from tests.core.h5sig import h5_content_signature

    out = write_reference_scan(str(tmp_path / "ref.nxs"),
                               str(tmp_path / "proj"))
    now = h5_content_signature(out)
    ref = json.loads(FIXTURE.read_text())

    missing = sorted(set(ref) - set(now))
    added = sorted(set(now) - set(ref))
    assert not missing and not added, (
        f"tree changed: missing={missing[:6]} added={added[:6]}"
    )
    diffs = [k for k in sorted(ref) if ref[k] != now[k]]
    assert diffs == [], (
        "content changed at: " + ", ".join(diffs[:8]) + "\n"
        + "\n".join(f"  {k}: ref={ref[k]} now={now[k]}" for k in diffs[:3])
    )


# ── C1/C2: storage-layout tripwire (the half h5_content_signature misses) ─────
# h5_content_signature digests VALUES + dtypes + attrs, not the on-disk storage
# layout — so a writer regression in chunking / compression / resizability slips
# past the content-identity gate above.  These additive tests pin that layout
# WITHOUT touching the frozen content signature or its committed fixture.

def test_v2_record_storage_layout_frozen(tmp_path):
    """The integrated stacks stay chunked + resizable (the streaming append
    depends on both); the label/axis columns keep their shape; and NOTHING
    re-emits raw lzf (the ARM64 bus-error codec — never re-emitted, see the
    avoid-lzf policy).  All mode-independent (holds for lz4/gzip/none)."""
    import h5py
    from tests.core._v2_record_fixture import write_reference_scan

    out = write_reference_scan(str(tmp_path / "ref.nxs"), str(tmp_path / "proj"))
    with h5py.File(out, "r") as f:
        # appendable integrated stacks: chunked + resizable along frame axis
        for p in ("entry/integrated_1d/intensity", "entry/integrated_1d/sigma",
                  "entry/integrated_2d/intensity"):
            ds = f[p]
            assert ds.chunks is not None, f"{p}: lost chunking"
            assert ds.maxshape[0] is None, f"{p}: not resizable (append broken)"
        # frame_index: chunked + resizable for append, but never compressed
        for p in ("entry/integrated_1d/frame_index",
                  "entry/integrated_2d/frame_index"):
            ds = f[p]
            assert ds.chunks is not None and ds.maxshape[0] is None
            assert ds.compression is None
        # fixed axes: not chunked, not compressed
        for p in ("entry/integrated_1d/q", "entry/integrated_2d/q",
                  "entry/integrated_2d/chi"):
            ds = f[p]
            assert ds.chunks is None and ds.compression is None
        # the ARM64 guard: no dataset re-emits raw lzf anywhere in the tree
        lzf = []
        f.visititems(lambda n, o: lzf.append(n) if isinstance(o, h5py.Dataset)
                     and o.compression == "lzf" else None)
        assert lzf == [], f"raw lzf re-emitted (ARM64 bus-error risk): {lzf}"


def test_v2_record_integrated_stacks_compress(tmp_path, monkeypatch):
    """The integrated stacks honor the resolved codec: ``gzip`` yields a portable
    (stock-h5py-readable) file; the default compresses (lz4, or a gzip fallback
    when hdf5plugin is absent).  Both keep chunking + resizability; neither
    re-emits raw lzf.  The GUI writer caches the resolved codec in the module
    constant ``INTEGRATED_STACK_COMPRESSION`` (the env var is read only at
    import), so we set THAT, not the env var."""
    import h5py
    import xdart.modules.ewald.nexus_writer as nw
    from xrd_tools.io.nexus import resolve_stack_compression
    from tests.core._v2_record_fixture import write_reference_scan

    stacks = ("entry/integrated_1d/intensity", "entry/integrated_1d/sigma",
              "entry/integrated_2d/intensity")
    # portable gzip
    monkeypatch.setattr(nw, "INTEGRATED_STACK_COMPRESSION", "gzip")
    out = write_reference_scan(str(tmp_path / "g.nxs"), str(tmp_path / "gp"))
    with h5py.File(out, "r") as f:
        for p in stacks:
            assert f[p].compression == "gzip", f"{p}: not portable gzip"
            assert f[p].chunks is not None and f[p].maxshape[0] is None
    # default: compressed (lz4, or gzip when hdf5plugin missing), never raw lzf
    monkeypatch.setattr(nw, "INTEGRATED_STACK_COMPRESSION",
                        resolve_stack_compression("lz4"))
    out = write_reference_scan(str(tmp_path / "d.nxs"), str(tmp_path / "dp"))
    with h5py.File(out, "r") as f:
        for p in stacks:
            assert f[p].compression not in (None, "lzf"), f"{p}: lost default compression"
