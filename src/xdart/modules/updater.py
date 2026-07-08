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

PYPI_JSON_URL = "https://pypi.org/pypi/xrd-tools/json"
_META_FILENAME = "install_meta.json"


def current_version():
    """The running xrd-tools version, or None if metadata is unavailable."""
    try:
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
    """True if xrd-tools is an editable/development checkout, via the dist-info
    ``direct_url.json`` ``dir_info.editable`` flag (PEP 610/660).  Never raises."""
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


def install_kind(meta=None):
    """Classify the install for update handling:

    * ``"editable"`` -- a dev checkout; refuse (use git).
    * ``"pixi"``     -- installer ``install_meta.json`` present; update-on-exit.
    * ``"managed"``  -- pip/conda, no meta; show a copyable command, never auto-run.
    """
    if is_editable_install():
        return "editable"
    resolved = meta if meta is not None else find_install_meta()
    return "pixi" if resolved is not None else "managed"


def fetch_latest_pypi(*, timeout=3.0, opener=None):
    """Latest xrd-tools version string from PyPI, or None on ANY failure (offline,
    timeout, malformed JSON).  Never raises.  ``opener`` is injectable for tests:
    a callable ``(url, timeout) -> bytes``.
    """
    try:
        if opener is not None:
            raw = opener(PYPI_JSON_URL, timeout)
        else:
            from urllib.request import urlopen
            with urlopen(PYPI_JSON_URL, timeout=timeout) as resp:
                raw = resp.read()
        data = json.loads(raw)
    except Exception:
        return None
    version = data.get("info", {}).get("version") if isinstance(data, dict) else None
    return str(version) if version else None


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
