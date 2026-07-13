# -*- coding: utf-8 -*-
"""One shared "last browsed directory" for every Browse button.

Rule (maintainer, 2026-07-13): a file dialog opens in the directory of the
LAST successful pick — any field, any wrangler, any mode — persisted across
sessions (``session.json`` key ``last_browse_dir``).  Fallbacks: the current
field's own folder, then the caller's fallback (e.g. the Project Folder),
then Qt's default.  Every dialog SEEDS from :func:`browse_start_dir` and
RECORDS its pick with :func:`remember_browse_path`; the rule lives here only.

Qt-free on purpose: dialogs stay at the call sites, persistence rides the
existing merge-on-write :mod:`xdart.utils.session` store.
"""

import logging
import os

from xdart.utils.session import load_session, save_session

logger = logging.getLogger(__name__)

_SESSION_KEY = 'last_browse_dir'


def _existing_dir_of(path) -> str:
    """The path itself when it is a directory, else its parent — '' if neither
    exists (a vanished directory must never seed a dialog)."""
    p = str(path or '').strip()
    if not p:
        return ''
    d = p if os.path.isdir(p) else os.path.dirname(p)
    return d if d and os.path.isdir(d) else ''


def last_browse_dir() -> str:
    """The persisted last-picked directory, '' when unset or vanished."""
    try:
        return _existing_dir_of(load_session().get(_SESSION_KEY))
    except Exception:
        logger.debug("last_browse_dir read failed", exc_info=True)
        return ''


def remember_browse_path(path) -> None:
    """Record a successful pick (file or directory) as the shared last dir."""
    d = _existing_dir_of(path)
    if d:
        save_session({_SESSION_KEY: d})


def browse_start_dir(current='', fallback='') -> str:
    """Start directory for a file dialog: last pick > current field's folder
    > ``fallback`` (expanded) > '' (Qt decides)."""
    last = last_browse_dir()
    if last:
        return last
    cur = _existing_dir_of(current)
    if cur:
        return cur
    return _existing_dir_of(os.path.expanduser(str(fallback or '').strip()))


def suggest_save_path(filename) -> str:
    """A save-dialog seed: the suggested ``filename`` inside the start dir
    (bare ``filename`` when no start dir is known — Qt then decides)."""
    start = browse_start_dir()
    name = str(filename or '')
    return os.path.join(start, name) if start else name
