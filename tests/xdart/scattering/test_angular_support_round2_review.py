"""Round-2 review: a selection already held retires an obsolete reconstruction."""
import threading
from types import SimpleNamespace

from tests.xdart.test_latest_request import qapp, _settled  # noqa: F401


def test_returning_to_completed_selection_cancels_obsolete_job(qapp, monkeypatch):
    from xdart.gui.tabs.scattering import angular_support_owner as module
    from xrd_tools.analysis.angular_support import SavedSupportRequest
    from xrd_tools.core.angular_support import AngularSupportProfile, SupportReason

    started, release = threading.Event(), threading.Event()
    cancellation = []
    profile = AngularSupportProfile.unavailable(SupportReason.LEGACY_INSUFFICIENT)

    def job(request, cancel, progress):
        if request.selected == (2,):
            cancellation.append(cancel)
            started.set()
            assert release.wait(10)
        return {label: profile for label in request.selected}

    monkeypatch.setattr(module, "_reconstruct", job)
    source = SimpleNamespace(path="one", entry="entry", snapshot=None, labels=(1, 2))
    source.request = lambda labels: SavedSupportRequest("one", "entry", None, (1, 2), tuple(labels))
    owner = module.AngularSupportOwner()
    try:
        owner.profiles(source, (1,))
        _settled(owner._requests)
        owner.profiles(source, (1, 2))
        assert started.wait(10)
        assert owner.profiles(source, (1,)) == {1: profile}
        assert cancellation[0].is_set(), "the no-longer-shown job should be cancelled immediately"
        assert not owner.computing
    finally:
        release.set()
        _settled(owner._requests)
        assert owner.shutdown().status.name == "CLEAN"
        owner.deleteLater()
