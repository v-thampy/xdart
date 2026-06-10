"""C4: the runtime min-ssrl-version guard (xdart_main.MIN_SSRL_VERSION).

The pip floor in pyproject.toml only protects pip installs — the documented
dev workflow is an editable install from a sibling clone, which bypasses it.
The startup guard turns "crashes on the first write" into a clear error, and
this test pins the guard's constant to the pyproject floor so they cannot
drift apart.
"""
import pathlib
import re
import tomllib


def _pyproject_floor() -> str:
    data = tomllib.loads(
        (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())
    for dep in data["project"]["dependencies"]:
        m = re.match(r"ssrl_xrd_tools\s*>=\s*([0-9][0-9.]*)", dep)
        if m:
            return m.group(1)
    raise AssertionError("no ssrl_xrd_tools floor in pyproject.toml")


def test_min_version_matches_pyproject_floor():
    from xdart.xdart_main import MIN_SSRL_VERSION
    assert MIN_SSRL_VERSION == _pyproject_floor()


def test_capability_probe_passes_on_current_ssrl():
    # The sibling ssrl checkout must provide every load-bearing symbol the
    # probe checks — if this fails, xdart and ssrl have genuinely drifted.
    from xdart.xdart_main import _ssrl_capabilities_ok
    assert _ssrl_capabilities_ok()


def test_check_passes_on_current_install():
    # Must not raise: either the installed version satisfies the floor, or
    # the capability probe covers a stale editable-install version stamp.
    from xdart.xdart_main import check_ssrl_version
    check_ssrl_version()


def test_version_tuple_comparisons():
    from xdart.xdart_main import _version_tuple
    assert _version_tuple("0.41.0") > _version_tuple("0.40.0")
    assert _version_tuple("0.41.0") == (0, 41, 0)
    assert _version_tuple("1.2") == (1, 2, 0)
    assert _version_tuple("0.0.0+unknown") == (0, 0, 0)
    assert _version_tuple("0.41.1") > _version_tuple("0.41.0")
