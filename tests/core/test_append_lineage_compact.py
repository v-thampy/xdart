"""C-12 of NEXT-1210-RESPONSIVENESS: a multi-epoch Append history saves each epoch's new images only.

The maintainer, 2026-11-12 (verbatim): "Add it to packet C now"; "Save only each epoch's new images (Recommended)";
and on compatibility with older builds: "Don't worry about V1. 2.9 yet. It's barely been used so far. Proceed like that
is not a factor". Measured before (leaves c3-sample-*, c12-before-timing-*): each Live epoch stored its full source, so
epoch k held all k images (4,308,620 bytes at 200 single-frame epochs) and every epoch re-validated all of them.

Version 2 (two or more epochs; one epoch stays version 1, byte for byte): each epoch keeps its source's scalars and
only the members and dataset paths it added or grew, with counts of what it inherits from the epoch before. The oracle
is version 1's own validator: a version-2 history is accepted only when its expansion to version 1 passes the
version-1 rules and recompacts to itself; a tampered one is refused.
"""
from __future__ import annotations

import copy
import json

import pytest

from xrd_tools.io import append
from xrd_tools.io.append import (
    AppendExternalMember,
    AppendImageMember,
    AppendIntent,
    AppendSource,
    begin_same_run_lineage,
    extend_same_run_lineage,
    seal_append_epoch,
    truncate_append_epoch,
)


def _images(stop, *, start=0, size=1000):
    return tuple(AppendImageMember(path=f"/data/s_{i:04d}.tif", size=size, mtime_ns=10_000 + i, source_start=i,
                                   source_stop=i + 1, ordinal=i) for i in range(start, stop))


def _image_source(extent, generation, *, digest=None, size=0):
    return AppendSource(path="/data", adapter_id="tiff_series", size=size, mtime_ns=0, extent=extent, digest=digest,
                        image_members=_images(extent), generation=generation)


def _container_source(stops, generation):
    """A container whose members end at ``stops``; its last member may still grow."""
    members, start = [], 0
    for ordinal, stop in enumerate(stops):
        members.append(AppendExternalMember(path=f"/data/part_{ordinal}.h5", dataset_path="/entry/data",
                                            size=100 * (stop - start), mtime_ns=20_000 + stop, source_start=start,
                                            source_stop=stop, ordinal=ordinal))
        start = stop
    return AppendSource(path="/data/master.h5", adapter_id="nexus_hdf5", size=500 + stops[-1], mtime_ns=30_000 + stops[-1],
                        extent=stops[-1], dataset_paths=tuple(f"/entry/data_{i}" for i in range(len(stops))),
                        external_members=tuple(members), generation=generation)


def _intent(source):
    return AppendIntent("entry", "/data", "scan-identity", "science", ("1d",), source, tuple(range(source.extent)))


def _grow(sources, *, replace_pending=()):
    """The Live path: begin, then extend and seal one epoch per source; ``replace_pending`` sources extend the pending
    epoch before the next one is sealed."""
    intent = _intent(sources[0])
    decision = seal_append_epoch(begin_same_run_lineage(intent))
    for index, source in enumerate(sources[1:], start=1):
        current = _intent(source)
        decision = extend_same_run_lineage(decision, intent, current)
        intent = current
        for later in replace_pending if index == len(sources) - 1 else ():
            current = _intent(later)
            decision = extend_same_run_lineage(decision, intent, current)
            intent = current
        assert decision.disposition is append.AppendDisposition.WRITE, decision.reason
        decision = seal_append_epoch(decision)
    return decision.lineage


def _version_one(sources):
    """The version-1 history of the same epochs: each epoch's complete source and its new labels."""
    epochs, done = [], 0
    for source in sources:
        epochs.append({"source": append._source_dict(source), "labels": list(range(done, source.extent))})
        done = source.extent
    return {"version": 1, "state": "committed", "entry": "entry", "source_base": append._normalize("/data"),
            "source_identity": "scan-identity", "science_fingerprint": "science", "modes": ["1d"], "epochs": epochs}


def _stored_members(lineage):
    return sum(len(epoch["source"]["image_members"]) + len(epoch["source"]["external_members"])
               for epoch in lineage["epochs"])


FIXTURES = {
    "one frame per epoch": [_image_source(n, n) for n in range(1, 31)],
    "batches": [_image_source(n, g) for g, n in enumerate((3, 8, 9, 15, 15, 21), start=1)],
    "growing container": [_container_source(stops, g) for g, stops in enumerate(
        ((5,), (8,), (8, 12), (8, 12, 13), (8, 12, 20)), start=1)],
}


@pytest.mark.parametrize("name", tuple(FIXTURES))
def test_a_multi_epoch_history_saves_each_epochs_new_members_and_expands_to_version_one(name):
    sources = FIXTURES[name]
    lineage = _grow(sources)
    images = sum(len(source.image_members) for source in sources)
    assert lineage["version"] == 2, "a multi-epoch history is compact"
    final = sources[-1]
    assert _stored_members(lineage) <= len(final.image_members) + len(final.external_members) + len(sources), (
        _stored_members(lineage), images)  # each member once (a growing container's last member once per growth)
    expanded = append._expand_lineage(lineage)
    assert expanded == _version_one(sources)
    assert append._lineage_labels(expanded) == append._lineage_labels(lineage) == tuple(range(final.extent))
    assert append._compact_lineage(expanded) == lineage  # canonical: the expansion recompacts to itself
    assert append._lineage_final_source(lineage) == append._source_dict(final)
    assert [dict(state) for state in append.lineage_epoch_states(lineage)] == [
        {**{key: value for key, value in append._source_dict(source).items()
            if key not in ("image_members", "external_members", "dataset_paths")},
         "member_count": len(source.image_members) + len(source.external_members)} for source in sources]


def test_one_epoch_stays_version_one_byte_for_byte():
    source = _image_source(4, 1)
    lineage = seal_append_epoch(begin_same_run_lineage(_intent(source))).lineage
    assert lineage == _version_one([source])


def test_the_history_grows_linearly_with_the_frames():
    """200 single-frame epochs: version 1 stored 20,100 image entries; version 2 stores each once."""
    lineage = _grow([_image_source(n, n) for n in range(1, 201)])
    assert _stored_members(lineage) == 200
    assert len(append._json(lineage)) < 200 * 400


def test_a_pending_epoch_replaced_twice_and_a_stop_truncated_epoch_expand_to_version_one():
    sources = [_image_source(n, n) for n in (2, 4)]
    lineage = _grow(sources, replace_pending=[_image_source(6, 5), _image_source(9, 6)])
    assert append._expand_lineage(lineage) == _version_one([sources[0], _image_source(9, 6)])
    prior = _intent(_image_source(4, 4))
    decision = seal_append_epoch(extend_same_run_lineage(
        seal_append_epoch(begin_same_run_lineage(prior)), prior, _intent(_image_source(10, 10))))
    pending = extend_same_run_lineage(decision, _intent(_image_source(10, 10)), _intent(_image_source(14, 14)))
    truncated, _intent_after = truncate_append_epoch(pending, _intent(_image_source(14, 14)), (10, 11))
    sealed = seal_append_epoch(truncated).lineage
    expected = _version_one([_image_source(4, 4), _image_source(10, 10), append.truncate_append_source(
        _image_source(14, 14), 12)])
    assert append._expand_lineage(sealed) == expected
    assert append._compact_lineage(expected) == sealed


def test_a_version_one_multi_epoch_history_is_extended_as_version_two():
    sources = [_image_source(n, n) for n in (2, 5, 7)]
    old = _version_one(sources)
    prior = _intent(sources[-1])
    decision = append._decision(append.AppendDisposition.WRITE, (), tuple(range(7)), lineage=old)
    extended = seal_append_epoch(extend_same_run_lineage(decision, prior, _intent(_image_source(9, 9)))).lineage
    assert extended["version"] == 2
    assert append._expand_lineage(extended) == _version_one([*sources, _image_source(9, 9)])


def _set(path, value):
    def tamper(lineage):
        node = lineage
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return tamper


def _drop(path):
    def tamper(lineage):
        node = lineage
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return tamper


def _also(*tampers):
    def tamper(lineage):
        for each in tampers:
            each(lineage)
    return tamper


IMAGE_TAMPERS = {
    "inherited count is a bool": _set(("epochs", 2, "inherited_members"), True),
    "inherited count out of range": _set(("epochs", 2, "inherited_members"), 99),
    "inherited count short of the prior images": _set(("epochs", 2, "inherited_members"), 1),
    "first epoch inherits": _set(("epochs", 0, "inherited_members"), 1),
    "inherited dataset paths": _set(("epochs", 2, "inherited_dataset_paths"), 1),
    "epoch key missing": _drop(("epochs", 2, "inherited_dataset_paths")),
    "epoch key extra": _set(("epochs", 2, "extra"), 0),
    "member ordinal": _set(("epochs", 2, "source", "image_members", 0, "ordinal"), 7),
    "member range": _set(("epochs", 2, "source", "image_members", 0, "source_start"), 1),
    "member path not normal": _set(("epochs", 2, "source", "image_members", 0, "path"), "/data/../data/s_0002.tif"),
    "member type": _set(("epochs", 2, "source", "image_members", 0, "size"), "1000"),
    "member bool": _set(("epochs", 2, "source", "image_members", 0, "size"), True),
    "member key extra": _set(("epochs", 2, "source", "image_members", 0, "extra"), 1),
    "member dropped": _drop(("epochs", 2, "source", "image_members", 0)),
    "extent beyond members": _set(("epochs", 2, "source", "extent"), 4),
    "extent regressed": _also(_set(("epochs", 2, "source", "extent"), 1),
                              _set(("epochs", 2, "source", "image_members"), [])),
    "size regressed": _set(("epochs", 2, "source", "size"), -1),
    "mtime regressed": _also(_set(("epochs", 1, "source", "mtime_ns"), 9), _set(("epochs", 2, "source", "mtime_ns"), 5)),
    "digest downgraded": _set(("epochs", 1, "source", "digest"), "abc"),
    "path changed": _set(("epochs", 2, "source", "path"), "/elsewhere"),
    "adapter changed": _set(("epochs", 2, "source", "adapter_id"), "image_file"),
    "source key extra": _set(("epochs", 2, "source", "extra"), 1),
    "labels gapped": _set(("epochs", 2, "labels"), [3]),
    "labels overlap": _set(("epochs", 2, "labels"), [1, 2]),
}


@pytest.mark.parametrize("name", tuple(IMAGE_TAMPERS))
def test_a_tampered_compact_history_is_refused(name):
    lineage = _grow([_image_source(n, n) for n in (1, 2, 3, 4)])
    assert lineage["version"] == 2
    broken = copy.deepcopy(lineage)
    IMAGE_TAMPERS[name](broken)
    with pytest.raises((ValueError, TypeError, KeyError)):
        append._lineage_labels(json.loads(json.dumps(broken)))


CONTAINER_TAMPERS = {
    "grown member changed path": _set(("epochs", 1, "source", "external_members", 0, "path"), "/data/other.h5"),
    "grown member shrank": _also(_set(("epochs", 1, "source", "external_members", 0, "source_stop"), 4),
                                 _set(("epochs", 1, "source", "extent"), 4)),
    "dataset paths not covering members": _set(("epochs", 2, "source", "dataset_paths"), []),
}


@pytest.mark.parametrize("name", tuple(CONTAINER_TAMPERS))
def test_a_tampered_compact_container_history_is_refused(name):
    lineage = _grow(FIXTURES["growing container"])
    broken = copy.deepcopy(lineage)
    CONTAINER_TAMPERS[name](broken)
    with pytest.raises((ValueError, TypeError, KeyError)):
        append._lineage_labels(json.loads(json.dumps(broken)))


def test_a_one_epoch_version_two_history_is_refused():
    lineage = _grow([_image_source(n, n) for n in (1, 2)])
    lineage["epochs"] = lineage["epochs"][:1]
    lineage["epochs"][0]["labels"] = [0]
    with pytest.raises(ValueError):
        append._lineage_labels(lineage)
