# -*- coding: utf-8 -*-
"""In-app updater support (Help -> Check for Updates...).

Pure, Qt-free logic: install-flavor discovery, version comparison, and the PyPI
latest-version fetch.  The GUI (``xdart._gui_main``) wraps these on a worker
thread and drives the update-on-exit helper (``xdart._updater``).  Every function
here is deterministic and dependency-injectable so it can be tested with no Qt, no
network, and no real package install.  See
``docs/design/design_install_and_update_jul2026.md`` section 4.
"""
from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as _pkg_version
from pathlib import Path

PYPI_JSON_URL = "https://pypi.org/pypi/xdart/json"
#: The conda channel xdart is published to; used when a pixi-global manifest names no URL channel.
DEFAULT_CONDA_CHANNEL = "https://prefix.dev/xrd-tools"
_META_FILENAME = "install_meta.json"


def current_version():
    """The running xdart version, or None if metadata is unavailable."""
    try:
        return _pkg_version("xdart")
    except PackageNotFoundError:
        try:
            # pre-1.0 conda builds shipped under the old distribution name
            return _pkg_version("xrd-tools")
        except PackageNotFoundError:
            return None


def _load_meta(path):
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_install_meta(*, prefix=None, env_override=None):
    """Locate + parse the installer's ``install_meta.json``.

    Discovery order: ``XDART_INSTALL_META`` (explicit; a dir or the file itself),
    else walk UP from ``sys.prefix`` (the pixi env lives under
    ``APP_ROOT/.pixi/envs/...`` while the meta file sits at ``APP_ROOT``).  Returns
    the parsed dict, or None if nothing is found / it is unreadable.
    """
    override = (env_override if env_override is not None
                else os.environ.get("XDART_INSTALL_META"))
    if override:
        p = Path(override)
        return _load_meta(p if p.name == _META_FILENAME else p / _META_FILENAME)
    start = Path(prefix if prefix is not None else sys.prefix)
    try:
        start = start.resolve()
    except OSError:
        pass
    for directory in (start, *start.parents):
        meta = _load_meta(directory / _META_FILENAME)
        if meta is not None:
            return meta
    return None


def is_editable_install():
    """True if xdart is an editable/development checkout, via the dist-info
    ``direct_url.json`` ``dir_info.editable`` flag (PEP 610/660).  Never raises."""
    try:
        raw = distribution("xdart").read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        try:
            raw = distribution("xrd-tools").read_text("direct_url.json")
        except (PackageNotFoundError, OSError):
            return False
    if not raw:
        return False
    try:
        info = json.loads(raw)
    except ValueError:
        return False
    return bool(info.get("dir_info", {}).get("editable", False))


def _pixi_home():
    return os.environ.get("PIXI_HOME") or os.path.join(
        os.path.expanduser("~"), ".pixi")


def pixi_global_home(prefix=None):
    """The pixi home a ``pixi global`` env lives in: ``sys.prefix`` is ``<home>/envs/<env>``.  It comes from the
    running env, never from ``$PIXI_HOME`` alone, so an install under another home updates that home."""
    try:
        return Path(prefix if prefix is not None else sys.prefix).resolve().parent.parent
    except OSError:
        return Path(prefix if prefix is not None else sys.prefix).parent.parent


def pixi_global_env_name(prefix=None):
    """The ``pixi global`` environment name (the last part of ``sys.prefix``)."""
    try:
        return Path(prefix if prefix is not None else sys.prefix).resolve().name
    except OSError:
        return Path(prefix if prefix is not None else sys.prefix).name


def detect_pixi_global_env(prefix=None):
    """True if the running interpreter is a ``pixi global`` env -- ``sys.prefix``
    is ``<PIXI_HOME>/envs/<name>`` (parent dir ``envs``, grandparent the
    ``.pixi``/``PIXI_HOME`` root).  ``pixi global`` installs carry no
    ``install_meta.json`` (pixi owns the env), so they need this path probe."""
    try:
        p = Path(prefix if prefix is not None else sys.prefix).resolve()
    except OSError:
        return False
    if p.parent.name != "envs":
        return False
    root = p.parent.parent
    home = os.environ.get("PIXI_HOME")
    if home:
        try:
            if root == Path(home).resolve():
                return True
        except OSError:
            pass
    return root.name == ".pixi"


def pixi_global_meta(prefix=None, *, platform=None):
    """Updater meta for a ``pixi global`` install (no ``install_meta.json``): ``pixi global update <env>`` run with
    ``PIXI_HOME`` set to the env's own home, and a relaunch through that home's ``xdart`` trampoline (it sets the
    environment the Qt/HDF5 libraries need; never the env's bare python).  The home's own paths come first:
    inside the activated env ``shutil.which("xdart")`` finds the env's ``bin/xdart`` script, not the trampoline.
    On Windows both ``.exe`` files must exist, else None (the caller shows the copyable command)."""
    import shutil
    platform = sys.platform if platform is None else platform
    windows = platform.startswith("win")
    exe = ".exe" if windows else ""
    home = pixi_global_home(prefix)
    env = pixi_global_env_name(prefix)
    pixi = home / "bin" / f"pixi{exe}"
    if not pixi.exists():
        found = shutil.which("pixi")
        pixi = Path(found) if found else Path(_pixi_home()) / "bin" / f"pixi{exe}"
    trampoline = home / "bin" / f"xdart{exe}"
    if windows and not (pixi.exists() and trampoline.exists()):
        return None
    return {"flavor": "pixi-global", "app_root": str(home), "env_name": env,
            "update_cmd": [str(pixi), "global", "update", env],
            "update_env": {"PIXI_HOME": str(home)},
            "relaunch_cmd": [str(trampoline)]}


def install_kind(meta=None):
    """Classify the install for update handling:

    * ``"editable"``    -- a dev checkout; refuse (use git).
    * ``"pixi"``        -- script-installer ``install_meta.json`` present (checked first: the workspace env lives
      under ``<APP_ROOT>/.pixi/envs/``, which the pixi-global path probe would also accept).
    * ``"pixi-global"`` -- ``pixi global install``ed; ``pixi global update``.
    * ``"managed"``     -- pip/conda, no meta; copyable command, never auto-run.
    """
    if is_editable_install():
        return "editable"
    resolved = meta if meta is not None else find_install_meta()
    if resolved is not None:
        return "pixi"
    return "pixi-global" if detect_pixi_global_env() else "managed"


def resolve_update_meta(kind=None):
    """The update-on-exit meta (update_cmd/relaunch_cmd/app_root) for the given
    install kind, or None for editable/managed (no in-app auto-update), and for a
    pixi-global install on Windows whose pixi or trampoline executable is missing."""
    kind = kind or install_kind()
    if kind == "pixi-global":
        return pixi_global_meta()
    if kind == "pixi":
        return find_install_meta()
    return None


def _fetch(url, timeout, opener):
    """GET ``url`` with a named User-Agent: prefix.dev answers a channel file with a redirect to a signed download URL
    that refuses Python's default agent (HTTP 403, measured 2026-10-27).  Redirects are followed, never cached."""
    if opener is not None:
        return opener(url, timeout)
    from urllib.request import Request, urlopen
    agent = f"xdart/{current_version() or 'unknown'} (update check)"
    with urlopen(Request(url, headers={"User-Agent": agent}), timeout=timeout) as resp:
        return resp.read()


def fetch_latest_pypi(*, timeout=3.0, opener=None):
    """Latest xdart version string from PyPI, or None on ANY failure (offline,
    timeout, malformed JSON).  Never raises.  ``opener`` is injectable for tests:
    a callable ``(url, timeout) -> bytes``.
    """
    try:
        data = json.loads(_fetch(PYPI_JSON_URL, timeout, opener))
    except Exception:
        return None
    version = data.get("info", {}).get("version") if isinstance(data, dict) else None
    return str(version) if version else None


def latest_from_repodata(data, *, include_prereleases=False):
    """The highest ``xdart`` version in a conda channel's ``repodata.json`` (``packages`` and ``packages.conda``), as
    the channel spells it; pre-releases only when asked.  None when the channel lists no usable xdart build."""
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:                                  # pragma: no cover - packaging ships with the GUI extra
        return None
    best = None
    for key in ("packages", "packages.conda"):
        records = data.get(key) if isinstance(data, dict) else None
        if not isinstance(records, dict):
            continue
        for record in records.values():
            if not isinstance(record, dict) or record.get("name") != "xdart":
                continue
            raw = str(record.get("version") or "")
            try:
                version = Version(raw)
            except InvalidVersion:
                continue
            if version.is_prerelease and not include_prereleases:
                continue
            if best is None or version > best[0]:
                best = (version, raw)
    return None if best is None else best[1]


def pixi_global_channels(env_name, home):
    """The URL channels (``https://…``, ``file://…``) a pixi-global env installs from, in manifest order.  Bare names
    such as ``conda-forge`` are skipped (no xdart; a very large index).  Falls back to :data:`DEFAULT_CONDA_CHANNEL`."""
    try:
        import tomllib
        with open(Path(home) / "manifests" / "pixi-global.toml", "rb") as fh:
            channels = tomllib.load(fh)["envs"][env_name]["channels"]
    except Exception:
        return [DEFAULT_CONDA_CHANNEL]
    urls = []
    for channel in channels if isinstance(channels, list) else ():
        if isinstance(channel, dict):
            channel = channel.get("channel")
        if isinstance(channel, str) and "://" in channel:
            urls.append(channel.rstrip("/"))
    return urls or [DEFAULT_CONDA_CHANNEL]


def fetch_latest_conda(channels, *, timeout=3.0, opener=None, include_prereleases=False):
    """The latest xdart in the first of ``channels`` whose noarch index lists it, or None on any failure (never
    raises)."""
    for channel in channels:
        try:
            data = json.loads(_fetch(f"{channel.rstrip('/')}/noarch/repodata.json", timeout, opener))
        except Exception:
            continue
        latest = latest_from_repodata(data, include_prereleases=include_prereleases)
        if latest is not None:
            return latest
    return None


def fetch_latest_for(kind, *, timeout=3.0, opener=None, prefix=None):
    """The latest version for this install: a pixi-global install asks the conda channels it installs from (they may
    lag PyPI, and ``pixi global update`` can only reach what they hold); every other kind asks PyPI."""
    if kind == "pixi-global":
        current = current_version()
        try:
            from packaging.version import Version
            include = bool(current) and Version(current).is_prerelease
        except Exception:
            include = False
        channels = pixi_global_channels(pixi_global_env_name(prefix), pixi_global_home(prefix))
        return fetch_latest_conda(channels, timeout=timeout, opener=opener, include_prereleases=include)
    return fetch_latest_pypi(timeout=timeout, opener=opener)


def update_available(current, latest):
    """True iff ``latest`` is a strictly newer, non-pre-release upgrade over
    ``current``.  A stable user is never nudged onto a pre-release; a user already
    on a pre-release may move to a newer one.  Falls back to a plain string
    inequality only if ``packaging`` is somehow unavailable."""
    if not current or not latest:
        return False
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return latest != current and latest > current
    try:
        cur, new = Version(current), Version(latest)
    except InvalidVersion:
        return False
    if new.is_prerelease and not cur.is_prerelease:
        return False
    return new > cur
