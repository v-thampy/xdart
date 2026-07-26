"""Headless contracts for generation-stamped run configuration values."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.readiness import processing_config_from_mapping
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    FrozenSourceSpec,
    GIIntent,
    RunIntent,
    ThresholdIntent,
)
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent() -> RunIntent:
    nested_files = ["/data/scan_0001.tif", "/data/scan_0002.tif"]
    return RunIntent(
        source_spec=SourceSpec(
            Path("/data"),
            SourceKind.TIFF_SERIES,
            metadata_uri=Path("/data/scan.txt"),
            entry="entry",
            options={
                "files": nested_files,
                "selection": {"indices": [1, 2]},
            },
        ),
        processing_mode="Int 2D",
        output_mode="Replace",
        max_cores=4,
        bai_1d_args={
            "unit": "q_A^-1",
            "npt": 1000,
            "radial_range": [0.1, 5.0],
            "method": ("bbox", "csr", "cython"),
        },
        bai_2d_args={
            "unit": "q_A^-1",
            "npt_rad": 500,
            "npt_azim": 360,
            "nested": {"weights": [1.0, 2.0]},
        },
        gi=GIIntent(
            enabled=True,
            incidence_motor="halpha",
            th_val=0.12,
            sample_orientation=3,
            tilt_angle=0.25,
            mode_1d="q_ip",
            mode_2d="qip_qoop",
        ),
        threshold=ThresholdIntent(
            apply_threshold=True,
            threshold_min=10,
            threshold_max=60_000,
            mask_saturation=False,
        ),
        poni_file="/calibration/eiger.poni",
        poni_values={
            "dist": 0.2,
            "detector_config": {"pixel1": 75e-6},
        },
        mask_file="/calibration/eiger-mask.npy",
        project_root="/data",
        save_path="/processed",
        run_options={"writer": {"flush_every": 5}},
    )


def test_freeze_is_deeply_immutable_and_generation_is_content_independent():
    intent = _intent()
    frozen = intent.freeze()

    # Mutating every mutable owner-side input after Run cannot change this run.
    intent.bai_1d_args["radial_range"][0] = 99
    intent.bai_2d_args["nested"]["weights"].append(3.0)
    intent.gi.incidence_motor = "theta"
    intent.threshold.threshold_max = 20
    intent.poni_values["detector_config"]["pixel1"] = 1.0
    intent.run_options["writer"]["flush_every"] = 99
    intent.source_spec.options["files"].append("/data/scan_0003.tif")
    intent.source_spec.options["selection"]["indices"].append(3)

    assert frozen.generation == 1
    assert frozen.output_mode == "Overwrite"
    assert frozen.gi.incidence_motor == "halpha"
    assert frozen.threshold.threshold_max == 60_000
    assert frozen.bai_1d_args["radial_range"] == [0.1, 5.0]
    assert frozen.bai_2d_args["nested"]["weights"] == [1.0, 2.0]
    assert frozen.poni_values["detector_config"]["pixel1"] == 75e-6
    assert frozen.run_options["writer"]["flush_every"] == 5
    assert frozen.thaw_source_spec().options["files"] == [
        "/data/scan_0001.tif",
        "/data/scan_0002.tif",
    ]

    # Accessors return independent copies rather than exposing frozen storage.
    args = frozen.bai_1d_args
    args["radial_range"].append(12)
    provenance = frozen.as_provenance()
    provenance["bai_2d_args"]["nested"]["weights"].clear()
    assert frozen.bai_1d_args["radial_range"] == [0.1, 5.0]
    assert frozen.bai_2d_args["nested"]["weights"] == [1.0, 2.0]

    # Generation distinguishes accepted runs; fingerprint describes content.
    same_content = RunIntent.from_frozen(frozen).freeze()
    assert same_content.generation == 2
    assert same_content.fingerprint == frozen.fingerprint
    assert same_content.identity != frozen.identity


def test_accessors_preserve_gi_motor_and_processing_signature():
    frozen = _intent().freeze()

    assert frozen.scan_args() == {
        "bai_1d_args": frozen.bai_1d_args,
        "bai_2d_args": frozen.bai_2d_args,
    }
    kwargs = frozen.scan_kwargs()
    assert kwargs["gi"] is True
    assert kwargs["incidence_motor"] == "halpha"
    assert kwargs["skip_2d"] is False
    assert kwargs["apply_threshold"] is True
    assert kwargs["threshold_min"] == 10
    assert kwargs["threshold_max"] == 60_000
    assert kwargs["mask_sentinel"] is False

    mapping = frozen.processing_mapping()
    assert mapping["gi_config"] == {
        "gi_mode_1d": "q_ip",
        "gi_mode_2d": "qip_qoop",
        "incidence_motor": "halpha",
        "th_val": 0.12,
        "sample_orientation": 3,
        "tilt_angle": 0.25,
    }
    signature = processing_config_from_mapping(mapping)
    assert signature is not None
    assert signature.mode.value == "gi"
    assert signature.axis_1d == "q_ip"
    assert signature.axis_2d == "qip_qoop"
    assert signature.npt_1d == 1000
    assert signature.npt_rad_2d == 500
    assert signature.npt_azim_2d == 360


def test_manual_gi_motor_thaws_to_numeric_live_scan_value():
    intent = _intent()
    intent.gi.incidence_motor = "Manual"
    intent.gi.th_val = 0.375

    frozen = intent.freeze()

    assert frozen.gi.incidence_motor == "Manual"
    assert frozen.gi.scan_incidence_motor == "0.375"
    assert frozen.scan_kwargs()["incidence_motor"] == "0.375"
    assert frozen.processing_mapping()["gi_config"]["incidence_motor"] == "Manual"


def test_directory_source_round_trip_is_typed_and_independent(tmp_path):
    source = DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        suffixes=(".NXS", ".H5"),
        name_filter="sample",
        generation=8,
    )
    intent = RunIntent(source_spec=source)

    frozen = intent.freeze()
    intent.source_spec = None
    thawed = frozen.thaw_source_spec()

    assert isinstance(thawed, DirectorySourceSpec)
    assert thawed is not source
    assert thawed.root == tmp_path / "raw"
    assert thawed.recursive is True
    assert thawed.suffixes == (".nxs", ".h5")
    assert thawed.name_filter == "sample"
    assert thawed.generation == 8


def test_fingerprint_is_mapping_order_independent_and_value_sensitive():
    first = _intent()
    second = _intent()
    second.bai_1d_args = dict(reversed(tuple(second.bai_1d_args.items())))
    second.bai_2d_args = dict(reversed(tuple(second.bai_2d_args.items())))

    first_frozen = first.freeze()
    second_frozen = second.freeze()

    assert first_frozen.fingerprint == second_frozen.fingerprint
    second.gi.incidence_motor = "theta"
    changed = second.freeze()
    assert changed.fingerprint != second_frozen.fingerprint


def test_failed_freeze_does_not_consume_generation():
    intent = RunIntent(live_mode=True, batch_mode=True)

    with pytest.raises(
        ValueError,
        match="live_mode and batch_mode cannot both be enabled",
    ):
        intent.freeze()

    assert intent.generation == 0
    intent.batch_mode = False
    assert intent.freeze().generation == 1


def test_session_package_exposes_values_lazily_without_qt():
    from xrd_tools.session import RunIntent as PublicRunIntent

    assert PublicRunIntent is RunIntent
    assert issubclass(FrozenRunConfiguration, object)

    # R4B-13: import purity ("keeps xdart thin") must be proven in a FRESH
    # interpreter.  Asserting ``"pyqtgraph" not in sys.modules`` in THIS process
    # was order-fragile — any earlier Qt-importing test in the same pytest
    # process left pyqtgraph resident and failed the assertion vacuously.  A
    # subprocess isolates the fact under test: importing ``xrd_tools.session``
    # must not drag in Qt/pyqtgraph.
    import subprocess

    code = (
        "import sys\n"
        "import xrd_tools.session as session\n"
        "assert session.RunIntent is not None\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'qtpy' or name == 'pyqtgraph'\n"
        "    or name.startswith('qtpy.') or name.startswith('pyqtgraph.')\n"
        ")\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gi_motor_resolved_once_from_choices():
    """O-1a-ii item 5 (R4B-8): the effective GI motor is resolved ONCE at freeze
    from the supplied source-motor list, using the single shared policy — never
    injecting a motor absent from the source, and carrying both the raw
    selection and the resolved motor."""
    from xrd_tools.session.run_configuration import resolve_gi_motor

    # Policy directly.
    assert resolve_gi_motor("th", ["halpha", "detx"]) == "halpha"   # stale -> default
    assert resolve_gi_motor("halpha", ["halpha", "detx"]) == "halpha"  # valid kept
    assert resolve_gi_motor("detx", ["detx", "dety"]) == "detx"     # valid explicit kept
    assert resolve_gi_motor("Manual", ["halpha"]) == "Manual"       # Manual stays
    # A stale motor absent from the source with no rotation-like axis -> Manual.
    assert resolve_gi_motor("th", ["detx", "dety"]) == "Manual"
    assert resolve_gi_motor("th", None) == "th"                     # unverifiable -> as-is

    # GIIntent.freeze carries raw + resolved and drives the effective motor.
    frozen = GIIntent(enabled=True, incidence_motor="th", th_val=0.2).freeze(
        choices=["halpha", "detx"])
    assert frozen.incidence_motor == "th"          # raw selection retained
    assert frozen.resolved_motor == "halpha"       # resolved once
    assert frozen.effective_motor == "halpha"
    assert frozen.scan_incidence_motor == "halpha"
    assert frozen.scan_config()["incidence_motor"] == "halpha"
    assert frozen.as_dict()["resolved_motor"] == "halpha"

    # Manual resolves to the typed theta value for the scan.
    manual = GIIntent(enabled=True, incidence_motor="Manual", th_val=0.35).freeze(
        choices=["halpha"])
    assert manual.effective_motor == "Manual"
    assert manual.scan_incidence_motor == str(0.35)


def test_gi_resolved_motor_is_part_of_run_identity():
    """Two runs with the same RAW GI selection but a different RESOLVED motor
    (different source motor lists) have DIFFERENT fingerprints — the resolved
    axis is part of the run's content identity."""
    a = RunIntent(gi=GIIntent(enabled=True, incidence_motor="th")).freeze(
        gi_motor_choices=["halpha", "detx"])
    b = RunIntent(gi=GIIntent(enabled=True, incidence_motor="th")).freeze(
        gi_motor_choices=["eta", "detx"])
    assert a.gi.effective_motor == "halpha"
    assert b.gi.effective_motor == "eta"
    assert a.fingerprint != b.fingerprint


# --------------------------------------------------------------------------- #
# The two intrinsic source-value operations (review §43.1).
#
# ``filesystem_root`` and ``format_tokens`` are value semantics OWNED by the
# accepted frozen source: computed from one immutable value object, with no
# setter, cache, table, default or second authority.  They replace the worker
# side's legacy source wrappers, so this matrix is the contract those wrappers
# used to carry.  The freeze-owner end (which GUI card produces which shape) is
# asserted by the production-shaped GUI oracle, not here.
# --------------------------------------------------------------------------- #


def test_directory_source_values_preserve_every_frozen_answer(tmp_path):
    frozen = FrozenSourceSpec.from_source(DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        suffixes=(".TIF",),
        name_filter="sample",
    ))

    assert frozen.filesystem_root == str(tmp_path / "raw")
    assert frozen.format_tokens == ("tif",)
    assert frozen.recursive is True
    assert frozen.name_filter == "sample"


def test_container_directory_keeps_both_real_master_spellings(tmp_path):
    """An Eiger container directory freezes TWO spellings; both must survive.

    Collapsing ``("_master.hdf5", "_master.h5")`` to one token would make a
    consumer matching the other spelling silently discover nothing.
    """
    frozen = FrozenSourceSpec.from_source(DirectorySourceSpec(
        root=tmp_path,
        suffixes=("_master.hdf5", "_master.h5"),
    ))

    assert frozen.format_tokens == ("hdf5", "h5")
    assert frozen.filesystem_root == str(tmp_path)


def test_tiff_series_root_is_the_containing_directory(tmp_path):
    """The §40.1 P1-B grandparent bug, now owned by the value object."""
    from xrd_tools.sources import image_series_spec

    member = tmp_path / "scan_0001.tif"
    member.write_bytes(b"")
    frozen = FrozenSourceSpec.from_source(image_series_spec(member))

    assert frozen.source_kind == "tiff_series"
    assert frozen.filesystem_root == str(tmp_path)
    assert frozen.format_tokens == ("tif",)


@pytest.mark.parametrize(
    ("name", "kind", "token"),
    [
        ("frame.tif", SourceKind.IMAGE_FILE, "tif"),
        ("scan_master.h5", SourceKind.EIGER_MASTER, "h5"),
        ("scan_master.hdf5", SourceKind.EIGER_MASTER, "hdf5"),
        ("stack.nxs", SourceKind.NEXUS_STACK, "nxs"),
    ],
)
def test_file_shaped_source_root_is_the_parent(tmp_path, name, kind, token):
    frozen = FrozenSourceSpec.from_source(SourceSpec(tmp_path / name, kind))

    assert frozen.filesystem_root == str(tmp_path)
    assert frozen.format_tokens == (token,)


@pytest.mark.parametrize(
    "kind",
    [SourceKind.LIVE, SourceKind.TILED, SourceKind.MEMORY, SourceKind.UNKNOWN],
)
def test_non_filesystem_source_fabricates_no_root(kind):
    frozen = FrozenSourceSpec.from_source(SourceSpec("some://handle", kind))

    with pytest.raises(ValueError, match="no filesystem root"):
        frozen.filesystem_root


def test_absent_source_is_refused_rather_than_defaulted():
    frozen = RunIntent(source_spec=None).freeze()

    assert frozen.source is None
    assert frozen.thaw_source_spec() is None


def test_uri_suffix_fallback_never_reads_a_directory_name():
    """A dotted DIRECTORY name is not a format token.

    The uri-suffix fallback exists for a file-shaped source with no frozen
    suffix tuple; applying it to a source whose uri names a directory would
    reinvent exactly the inference §40.1 P1-B removed.
    """
    series = FrozenSourceSpec(
        family="source", uri="/data/scan.2026", source_kind="tiff_series")
    directory = FrozenSourceSpec(family="directory", uri="/data/scan.2026")

    assert series.format_tokens == ()
    assert directory.format_tokens == ()


def test_source_value_properties_stay_headless():
    """The value type and BOTH properties import no Qt and no xdart."""
    import subprocess

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from xrd_tools.core.scan import SourceKind, SourceSpec\n"
        "from xrd_tools.session.run_configuration import FrozenSourceSpec\n"
        "frozen = FrozenSourceSpec.from_source(\n"
        "    SourceSpec(Path('/data/scan_master.h5'), SourceKind.EIGER_MASTER))\n"
        "assert frozen.filesystem_root == str(Path('/data'))\n"
        "assert frozen.format_tokens == ('h5',)\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.split('.')[0] in {'qtpy', 'pyqtgraph', 'xdart'}\n"
        ")\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# The detached copy algebra (review §42.2, §43.3).
#
# ``_freeze_value`` accepts any object that can convert itself through ``item()``
# or ``tolist()``, so the copy boundary ``clone_candidate`` uses must cover that
# WHOLE algebra: a mutable container the freeze accepts but the copy aliases
# would let a later idle edit reach an already-published candidate.
# --------------------------------------------------------------------------- #


class _TolistOnly:
    """A MUTABLE value container the frozen algebra accepts via ``tolist``."""

    def __init__(self, items):
        self.items = list(items)

    def tolist(self):
        return list(self.items)


class _CopyReturnsSelf(_TolistOnly):
    """``copy()`` exists but hands back the same mutable object."""

    def copy(self):
        return self


class _ThrowingCopy(_TolistOnly):
    """``copy()`` exists and refuses; the value conversion still works."""

    def copy(self):
        raise ValueError("this container cannot copy itself")


class _HashableTolist(_TolistOnly):
    """Hash-sensitive: usable as a mapping key while still mutable."""

    def __hash__(self):
        return hash(tuple(self.items))

    def __eq__(self, other):
        return isinstance(other, _HashableTolist) and self.items == other.items


class _CopyableKey(_HashableTolist):
    """A hash-sensitive key that CAN hand back an independent copy."""

    def copy(self):
        return _CopyableKey(self.items)


@pytest.mark.parametrize(
    "factory", [_TolistOnly, _CopyReturnsSelf, _ThrowingCopy])
def test_clone_candidate_detaches_every_accepted_value_container(factory):
    value = factory([1, 2])
    intent = RunIntent(run_options={"probe": value})

    # The frozen algebra ACCEPTS this object, so the copy boundary owes it a
    # detached value; anything less contradicts the stated whole-algebra
    # contract rather than merely missing an exotic type.
    assert intent.freeze().run_options["probe"] == [1, 2]

    copied = intent.clone_candidate().run_options["probe"]

    assert copied is not value
    value.items.append(3)
    settled = copied.tolist() if hasattr(copied, "tolist") else copied
    assert settled == [1, 2]


def test_clone_candidate_detaches_or_refuses_a_hash_sensitive_mapping_key():
    """Keys ride the same boundary as values, or the copy fails closed.

    A key that can copy itself detaches; one that can only convert through
    ``tolist()`` becomes unhashable, and refusing is the correct fail-closed
    outcome -- returning the original key would alias it.
    """
    copyable = _CopyableKey([1, 2])
    intent = RunIntent(run_options={copyable: "value"})

    (copied_key,) = intent.clone_candidate().run_options
    assert copied_key is not copyable
    assert copied_key == copyable

    intent = RunIntent(run_options={_HashableTolist([3]): "value"})
    with pytest.raises(TypeError):
        intent.clone_candidate()


def test_clone_candidate_preserves_numpy_dtype_and_shape():
    numpy = pytest.importorskip("numpy")
    array = numpy.arange(4, dtype=numpy.int16).reshape(2, 2)
    intent = RunIntent(bai_1d_args={"weights": array})

    copied = intent.clone_candidate().bai_1d_args["weights"]

    assert copied is not array
    assert copied.dtype == array.dtype
    assert copied.shape == array.shape
    array[0, 0] = 99
    assert int(copied[0, 0]) == 0


def test_resolve_gi_motor_none_vs_empty_choices_escape():
    """The ()-vs-None distinction (T-1 escape fix): an explicit saved motor with
    UNKNOWN choices (``None``) must be HONORED (display/frozen must not diverge),
    while a genuinely EMPTY-and-known list (``()``) degrades a stale explicit
    motor to the default policy.  The GUI passes ``None`` when the dropdown is
    not populated — never ``()`` — so a motor GI run never silently becomes a
    fixed-angle run."""
    from xrd_tools.session.run_configuration import resolve_gi_motor

    # Unknown choices (None): the explicit motor is honored as-is.
    assert resolve_gi_motor("halpha", None) == "halpha"
    assert resolve_gi_motor("th", None) == "th"
    # Empty-and-known (()): a stale explicit motor degrades (no source motors).
    assert resolve_gi_motor("halpha", ()) == "Manual"
    assert resolve_gi_motor("th", ()) == "Manual"
    # Freeze honors an explicit motor when the caller cannot supply a list.
    frozen_unknown = GIIntent(enabled=True, incidence_motor="halpha").freeze(
        choices=None)
    assert frozen_unknown.effective_motor == "halpha"
    frozen_empty = GIIntent(enabled=True, incidence_motor="halpha").freeze(
        choices=())
    assert frozen_empty.effective_motor == "Manual"
