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
import time
import uuid
from dataclasses import dataclass
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
    exe = ".exe" if platform.startswith("win") else ""
    home = pixi_global_home(prefix)
    env = pixi_global_env_name(prefix)
    pixi = home / "bin" / f"pixi{exe}"
    if not pixi.exists():
        found = shutil.which("pixi")
        pixi = Path(found) if found else Path(_pixi_home()) / "bin" / f"pixi{exe}"
    # The relaunch command is the one this env's manifest exposes for xdart; an unreadable manifest, no exposed xdart
    # or a missing executable is an unknown state, so the caller shows the copyable command instead.
    name = exposed_xdart_command(pixi_global_entry(env, home))
    trampoline = home / "bin" / f"{name}{exe}" if name else None
    if trampoline is None or not trampoline.exists() or not pixi.exists():
        return None
    return {"flavor": "pixi-global", "app_root": str(home), "env_name": env, "env_prefix": str(home / "envs" / env),
            "update_cmd": [str(pixi), "global", "update", env],
            "update_env": {"PIXI_HOME": str(home)},
            "relaunch_cmd": [str(trampoline)]}


def pixi_global_command(words, *, prefix=None):
    """A ``pixi global …`` command line for this install's home, with ``PIXI_HOME`` when it is not the default."""
    home, default = pixi_global_home(prefix), Path(_pixi_home())
    command = f"pixi global {words}"
    try:
        same_home = home == default.resolve()
    except OSError:
        same_home = home == default
    if same_home:
        return command
    if sys.platform.startswith("win"):
        return f"$env:PIXI_HOME = '{home}'; {command}"
    return f'PIXI_HOME="{home}" {command}'


def copyable_update_command(kind, *, prefix=None):
    """The command a user can run by hand for this install: never pip for a pixi install."""
    if kind == "pixi-global":
        return pixi_global_command(f"update {pixi_global_env_name(prefix)}", prefix=prefix)
    if kind == "pixi":
        meta = find_install_meta() or {}
        cmd = meta.get("update_cmd")
        if isinstance(cmd, list) and cmd:
            return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
    return 'pip install -U "xdart[gui]"'


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


def pixi_global_entry(env_name, home):
    """The ``[envs.<env>]`` table of ``<home>/manifests/pixi-global.toml``, or None when it cannot be read."""
    try:
        import tomllib
        with open(Path(home) / "manifests" / "pixi-global.toml", "rb") as fh:
            entry = tomllib.load(fh)["envs"][env_name]
    except Exception:
        return None
    return entry if isinstance(entry, dict) else None


def _is_conda_forge(channel):
    return channel == "conda-forge" or channel.rstrip("/").endswith("/conda-forge")


def manifest_channels(entry):
    """``(urls, plain)``: the URL channels (``https://…``, ``file://…``) in priority order, skipping conda-forge (no
    xdart, a very large index); ``plain`` is False when a channel is neither a URL nor conda-forge, i.e. an authority
    this check cannot read."""
    channels = entry.get("channels") if isinstance(entry, dict) else None
    if not isinstance(channels, list) or not channels:
        return [], False
    urls, plain = [], True
    for channel in channels:
        if isinstance(channel, dict):
            plain = False
            channel = channel.get("channel")
        if not isinstance(channel, str) or not channel:
            plain = False
        elif _is_conda_forge(channel):
            continue
        elif "://" in channel:
            urls.append(channel.rstrip("/"))
        else:
            plain = False
    return urls, plain


def pixi_global_channels(env_name, home):
    """The URL channels a pixi-global env installs from, in manifest order; :data:`DEFAULT_CONDA_CHANNEL` when the
    manifest cannot be read (Help's manual check only; the automatic update requires a readable manifest)."""
    urls, _plain = manifest_channels(pixi_global_entry(env_name, home))
    return urls or [DEFAULT_CONDA_CHANNEL]


def xdart_spec_is_plain(entry):
    """True when the env asks for xdart without a version pin (``xdart = "*"``, or a table without a version)."""
    deps = entry.get("dependencies") if isinstance(entry, dict) else None
    spec = deps.get("xdart") if isinstance(deps, dict) else None
    if isinstance(spec, dict):
        if set(spec) - {"version"}:
            return False
        spec = spec.get("version", "*")
    return isinstance(spec, str) and spec.strip() in ("", "*")


def exposed_xdart_command(entry):
    """The exposed command name that runs xdart (``exposed = { xdart = "xdart" }``), preferring ``xdart``; None when
    the env exposes none, so there is no trampoline to relaunch through."""
    exposed = entry.get("exposed") if isinstance(entry, dict) else None
    if not isinstance(exposed, dict):
        return None
    names = sorted(name for name, target in exposed.items() if target == "xdart" and isinstance(name, str) and name)
    return "xdart" if "xdart" in names else (names[0] if names else None)


def query_conda_channels(channels, *, timeout=3.0, opener=None, include_prereleases=False):
    """``(status, latest)`` from the first channel whose noarch index lists xdart: ``("ok", version)``,
    ``("absent", None)`` when a channel answered but none lists xdart, ``("unreachable", None)`` when none answered.
    Never raises."""
    answered, unavailable = False, False
    for channel in channels:
        try:
            data = json.loads(_fetch(f"{channel.rstrip('/')}/noarch/repodata.json", timeout, opener))
        except Exception:
            # An unavailable higher-priority channel is unknown authority, not proof of package absence.
            unavailable = True
            continue
        answered = True
        latest = latest_from_repodata(data, include_prereleases=include_prereleases)
        if latest is not None:
            return ("unreachable" if unavailable else "ok"), latest
    return ("unreachable" if unavailable or not answered else "absent"), None


def fetch_latest_conda(channels, *, timeout=3.0, opener=None, include_prereleases=False):
    """The latest xdart in the first of ``channels`` whose noarch index lists it, or None."""
    return query_conda_channels(channels, timeout=timeout, opener=opener,
                                include_prereleases=include_prereleases)[1]


def _running_prerelease():
    try:
        from packaging.version import Version
        current = current_version()
        return bool(current) and Version(current).is_prerelease
    except Exception:
        return False


def fetch_latest_for(kind, *, timeout=3.0, opener=None, prefix=None):
    """The latest version for this install: a pixi-global install asks the conda channels it installs from (they may
    lag PyPI, and ``pixi global update`` can only reach what they hold); every other kind asks PyPI."""
    return check_for_update(kind, timeout=timeout, opener=opener, prefix=prefix).latest


@dataclass(frozen=True)
class UpdateCheck:
    """One update check. ``status``: ``ok``, ``absent`` (the channels answered without xdart), ``unreachable``, or
    ``unsupported`` (an install layout this updater does not read). ``automatic`` is True only when the startup
    update may act on it unattended: a validated plain pixi-global manifest, or the script installer's workspace
    within its major version (its manifest asks for ``>=1,<2``)."""

    status: str
    latest: str | None = None
    automatic: bool = False
    reason: str = ""


def check_for_update(kind, *, timeout=3.0, opener=None, prefix=None, meta=None):
    """Check the source this install updates from (see :class:`UpdateCheck`). Never raises."""
    if kind == "pixi-global":
        home, env = pixi_global_home(prefix), pixi_global_env_name(prefix)
        entry = pixi_global_entry(env, home)
        if entry is None:
            return UpdateCheck("unsupported", reason=f"cannot read {Path(home) / 'manifests' / 'pixi-global.toml'}")
        urls, plain_channels = manifest_channels(entry)
        if not urls:
            return UpdateCheck("unsupported", reason="the manifest names no channel URL for xdart")
        status, latest = query_conda_channels(urls, timeout=timeout, opener=opener,
                                              include_prereleases=_running_prerelease())
        plain = plain_channels and xdart_spec_is_plain(entry) and exposed_xdart_command(entry) is not None
        reason = "" if plain else "the manifest pins xdart, names another channel, or exposes no xdart command"
        return UpdateCheck(status, latest, status == "ok" and plain, reason)
    latest = fetch_latest_pypi(timeout=timeout, opener=opener)
    if latest is None:
        return UpdateCheck("unreachable")
    automatic = False
    if kind == "pixi":
        resolved = meta if meta is not None else find_install_meta()
        try:
            from packaging.version import Version
            automatic = (isinstance(resolved, dict) and isinstance(resolved.get("update_cmd"), list)
                         and isinstance(resolved.get("relaunch_cmd"), list)
                         and Version(latest).major == Version(current_version() or "0").major)
        except Exception:
            automatic = False
    return UpdateCheck("ok", latest, automatic)


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


# ── Per-install update state (attempts, the helper's handshake, failed targets) ──────────────────────────────────
#
# The app writes STATE_FILE before it starts the helper; only the helper writes HELPER_FILE (its PID, then the
# outcome).  A second xdart start reads both: a live helper means "being updated, quit"; a finished or dead helper
# settles the attempt by comparing versions (never exit codes: a pinned manifest exits 0 with nothing done).

STATE_FILE = "xdart-update-state.json"
HELPER_FILE = "xdart-update-helper.json"
JOB_FILE = "xdart-update-job.json"
LOCK_FILE = "xdart-update.lock"
ADMISSION_LOCK_FILE = "xdart-update-admission.lock"
LOG_FILE = "update.log"
#: The helper stops an update still running after this long (agreed with Codex N-2b0787f9).
UPDATE_TIME_LIMIT_S = 1200
#: How long the helper waits for xdart to close, and for pixi to finish after its child processes are stopped.
PARENT_WAIT_S = 60
STOP_GRACE_S = 30
#: Windows: how long programs still using the environment get to close once xdart has (pixi's xdart.exe launcher
#: exits just after xdart) before the update is refused.
ENV_WAIT_S = 10
#: The longest a helper can legitimately run (used only where the lock cannot be tested).
_HELPER_MAX_RUN_S = PARENT_WAIT_S + ENV_WAIT_S + UPDATE_TIME_LIMIT_S + STOP_GRACE_S + 60


class UpdateInProgress(RuntimeError):
    """This install already has an app admission or update helper."""


class _FileLock:
    def __init__(self, close):
        self._close = close

    def close(self):
        close, self._close = self._close, None
        if close is not None:
            close()


def take_file_lock(path, *, create=True):
    """Acquire one nonblocking OS lock, or None if owned. Unknown locking fails closed."""
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateFileW(str(path), 0xC0000000 if create else 0x80000000,
                                    0, None, 4 if create else 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            if error in (32, 33):
                return None
            raise ctypes.WinError(error)
        return _FileLock(lambda: kernel.CloseHandle(handle))
    import fcntl
    fh = open(path, "a+b" if create else "rb")
    try:
        fcntl.flock(fh.fileno(), (fcntl.LOCK_EX if create else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    except BaseException:
        fh.close()
        raise
    return _FileLock(fh.close)


def _file_lock_held(path):
    if not Path(path).exists():
        return False
    try:
        lock = take_file_lock(path, create=False)
        if lock is None:
            return True
        lock.close()
        return False
    except FileNotFoundError:
        return False
    except OSError:
        return None


def reserve_update(state_dir):
    """Bridge app admission to the exact helper handshake without another app overwriting it."""
    lock = take_file_lock(Path(state_dir) / ADMISSION_LOCK_FILE)
    if lock is None:
        raise UpdateInProgress("Another xdart update is starting for this install.")
    if helper_lock_held(state_dir) is not False:
        lock.close()
        raise UpdateInProgress("Another xdart update is running for this install.")
    if (read_json(Path(state_dir) / HELPER_FILE) or {}).get("state") == "blocked":
        lock.close()
        raise UpdateInProgress("The previous update could not prove cleanup; recover that update before retrying.")
    return lock


def helper_lock_held(state_dir):
    """True while an update helper for this install runs: the helper holds LOCK_FILE exclusively for its whole life,
    and the lock dies with the process, so a reused PID cannot fake it. None when the lock cannot be tested here."""
    return _file_lock_held(Path(state_dir) / LOCK_FILE)


def install_identity(prefix=None):
    """This install's identity for the state files: the resolved ``sys.prefix``."""
    try:
        return str(Path(prefix if prefix is not None else sys.prefix).resolve())
    except OSError:
        return str(prefix if prefix is not None else sys.prefix)


def update_state_dir(meta):
    """Where this install's update files live: ``<home>/xdart-update/<env>`` for pixi global (one home can hold
    several envs), ``<app_root>/xdart-update`` for the script installer's workspace."""
    if meta.get("flavor") == "pixi-global" and meta.get("env_name"):
        return Path(meta["app_root"]) / "xdart-update" / meta["env_name"]
    return Path(meta.get("app_root") or ".") / "xdart-update"


def write_json_atomic(path, data):
    """Write JSON through a temporary file, flushed to disk, then renamed over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_update_state(state_dir, install):
    """This install's state: ``{"install", "failed": [versions], "attempt": {...} | None}``."""
    data = read_json(Path(state_dir) / STATE_FILE)
    if data is None or data.get("install") != install:
        return {"install": install, "failed": [], "attempt": None}
    failed = data.get("failed")
    attempt = data.get("attempt")
    return {"install": install,
            "failed": sorted({str(v) for v in failed}) if isinstance(failed, list) else [],
            "attempt": attempt if isinstance(attempt, dict) else None}


def version_reached(current, target):
    """True when the running version is the target or newer (a newer manual install is not a failed attempt)."""
    try:
        from packaging.version import Version
        return Version(str(current)) >= Version(str(target))
    except Exception:
        return False


def begin_attempt(state_dir, *, install, current, target, log):
    """Record an attempt before the helper starts; returns the previous state for :func:`abandon_attempt`."""
    state = load_update_state(state_dir, install)
    previous = {**state, "failed": list(state["failed"])}
    state["attempt"] = {"from": current, "to": target, "log": str(log), "launched_at": time.time(),
                        "id": uuid.uuid4().hex, "authorized": False}
    write_json_atomic(Path(state_dir) / STATE_FILE, state)
    try:
        (Path(state_dir) / HELPER_FILE).unlink()
    except OSError:
        pass
    return previous


def abandon_attempt(state_dir, previous, attempt_id=None):
    """Undo :func:`begin_attempt` when the helper could not be started."""
    current = read_json(Path(state_dir) / STATE_FILE) or {}
    if attempt_id is None or (current.get("attempt") or {}).get("id") == attempt_id:
        write_json_atomic(Path(state_dir) / STATE_FILE, previous)


def job_attempt(job):
    if not isinstance(job.get("attempt_id"), str) or not job["attempt_id"]:
        return None
    state = load_update_state(job["state_dir"], job["install"])
    attempt = state["attempt"]
    return attempt if attempt is not None and attempt.get("id") == job.get("attempt_id") else None


def authorize_attempt(state_dir, install, attempt_id):
    state = load_update_state(state_dir, install)
    attempt = state["attempt"]
    if attempt is None or attempt.get("id") != attempt_id:
        raise UpdateInProgress("The update attempt was cancelled or replaced.")
    attempt["authorized"] = True
    write_json_atomic(Path(state_dir) / STATE_FILE, state)


def settle_previous_attempt(state_dir, *, install, current, alive):
    """At startup: ``("none" | "active" | "updated" | "incomplete", attempt)``.  ``active``: the helper is still
    running (the caller quits without touching the attempt).  Otherwise the attempt is settled by version, a missed
    target joins the failed targets the startup update never retries, and the attempt is cleared."""
    if (load_update_state(state_dir, install)["attempt"] is None
            and _file_lock_held(Path(state_dir) / ADMISSION_LOCK_FILE) is False):
        return "none", None
    reservation = take_file_lock(Path(state_dir) / ADMISSION_LOCK_FILE)
    if reservation is None:
        return "active", load_update_state(state_dir, install)["attempt"]
    try:
        return _settle_attempt(state_dir, install, current, alive)
    finally:
        reservation.close()


def _settle_attempt(state_dir, install, current, alive):
    state = load_update_state(state_dir, install)
    attempt = state["attempt"]
    if attempt is None:
        return "none", None
    helper = read_json(Path(state_dir) / HELPER_FILE) or {}
    if helper.get("attempt_id") not in (None, attempt.get("id")):
        helper = {}
    if helper.get("state") == "blocked":
        return "active", {**attempt, "helper": helper}   # lost custody is never proof of quiescence
    if helper.get("state") != "completed":
        held = helper_lock_held(state_dir)
        if held is None:      # no lock here (an unusual file system): the PID within the longest legitimate run
            pid = helper.get("pid")
            try:
                recent = time.time() - float(attempt.get("launched_at", 0)) < _HELPER_MAX_RUN_S
            except (TypeError, ValueError):
                recent = False
            held = recent and isinstance(pid, int) and pid > 0 and alive(pid)
        if held:
            return "active", attempt
    reached = version_reached(current, attempt.get("to"))
    if not reached and attempt.get("to"):
        state["failed"] = sorted(set(state["failed"]) | {str(attempt["to"])})
    state["attempt"] = None
    write_json_atomic(Path(state_dir) / STATE_FILE, state)
    try:
        (Path(state_dir) / HELPER_FILE).unlink()
    except OSError:
        pass
    return ("updated" if reached else "incomplete"), {**attempt, "helper": helper}


def target_failed(state_dir, install, version):
    """True when an earlier attempt at ``version`` did not complete (the startup update skips it)."""
    return str(version) in load_update_state(state_dir, install)["failed"]


def state_dir_for(kind, *, prefix=None):
    """This install's update folder (see :func:`update_state_dir`) without building the launch meta; None for kinds
    the in-app updater never runs."""
    if kind == "pixi-global":
        return pixi_global_home(prefix) / "xdart-update" / pixi_global_env_name(prefix)
    if kind == "pixi":
        meta = find_install_meta(prefix=prefix) if prefix is not None else find_install_meta()
        if isinstance(meta, dict) and meta.get("app_root"):
            return Path(meta["app_root"]) / "xdart-update"
    return None


def startup_update_notice():
    """Read-only bootstrap gate before loading Qt/native GUI libraries into an updating environment: None when xdart
    may start, else ``(title, text)`` saying why it does not (an update is starting or running, or the last update
    could not prove that its processes stopped and needs the recovery step the text names)."""
    try:
        root = state_dir_for(install_kind())
        if root is None:
            return None
        updating = ("xdart is updating",
                    "xdart is being updated in another window. It reopens by itself when the update has finished.")
        if _file_lock_held(root / ADMISSION_LOCK_FILE) is not False:
            return updating
        helper = read_json(root / HELPER_FILE) or {}
        if helper.get("state") == "blocked":
            attempt = load_update_state(root, install_identity())["attempt"] or {}
            log = attempt.get("log") or root / LOG_FILE
            return ("xdart update needs recovery",
                    "The last xdart update could not confirm that all its processes stopped, so xdart stays closed to "
                    f"protect its environment. See {log}. Once those processes have ended, rename the folder {root} "
                    "and start xdart again.")
        if helper.get("state") != "completed" and helper_lock_held(root) is not False:
            return updating
        return None
    except Exception:
        return None   # discovery failure never prevents an ordinary/editable installation from starting


def startup_update_blocked():
    """True when :func:`startup_update_notice` keeps xdart from starting."""
    return startup_update_notice() is not None
