from xdart.gui.tabs.static_scan.analysis_context import (
    AnalysisContext,
    PatternData,
)


def test_analysis_context_current_pattern_tuple_preserves_live_provider():
    ctx = AnalysisContext(
        current_pattern_provider=lambda: ([1, 2], [3, 4], "Q"),
        frame_labels_provider=lambda: [7])

    data = ctx.current_pattern()
    assert isinstance(data, PatternData)
    assert data.x == [1, 2]
    assert data.y == [3, 4]
    assert data.x_label == "Q"
    assert data.frame_label == "7"
    assert ctx.current_pattern_tuple() == ([1, 2], [3, 4], "Q")


def test_analysis_context_frame_pattern_and_scan_helpers_are_stable():
    seen = {}

    def frame_provider(idx):
        seen["idx"] = idx
        return ([idx], [idx + 1], "2θ")

    ctx = AnalysisContext(
        frame_pattern_provider=frame_provider,
        scan_uri_provider=lambda: "/tmp/scan.nxs",
        mask_provider=lambda uri: ("mask", uri),
        frame_labels_provider=lambda: ("a", "b"),
        metadata_provider=lambda: {"i0": 1.0})

    assert ctx.pattern_tuple_for_frame(5) == ([5], [6], "2θ")
    assert seen["idx"] == 5
    assert ctx.current_scan_uri() == "/tmp/scan.nxs"
    assert ctx.mask_for_scan_uri() == ("mask", "/tmp/scan.nxs")
    assert ctx.frame_labels() == ("a", "b")
    assert ctx.metadata() == {"i0": 1.0}


def test_analysis_context_bad_provider_data_returns_none():
    ctx = AnalysisContext(current_pattern_provider=lambda: ("only", "two"))
    assert ctx.current_pattern() is None
    assert ctx.current_pattern_tuple() is None
