import json
import logging
import os
import threading
from pathlib import Path

# A module-level Path instance is NOT safe to share: Python 3.12's Path
# lazily caches ``_str``/``_drv`` on first use, and that mutation races
# under concurrency (the batch worker + GUI thread both touching it), which
# manifests as a RecursionError deep in pathlib.  Keep the default as a
# plain string and build a fresh Path per call instead.
_DEFAULT = str(Path.home() / '.xdart' / 'session.json')


def _default_path() -> str:
    """Resolved per call: XDART_SESSION_FILE lets tests (and parallel
    instances) redirect persistence away from the real user session."""
    return os.environ.get('XDART_SESSION_FILE') or _DEFAULT
_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _is_fresh() -> bool:
    """``xdart -f`` starts a FRESH session: ignore any saved state on load and
    persist nothing on save (ephemeral), so the user's real saved session is
    neither read nor clobbered.  Only affects the DEFAULT path — an explicit
    ``path=`` (or ``-n NAME`` redirect) is honoured as given."""
    return bool(os.environ.get('XDART_SESSION_FRESH'))


def load_session(path=None) -> dict:
    if path is None and _is_fresh():
        return {}
    p = Path(path) if path else Path(_default_path())
    try:
        with _LOCK:
            return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        # Session persistence is a convenience — it must never crash a run.
        logger.debug("load_session failed", exc_info=True)
        return {}


def _json_default(obj):
    """Coerce numpy scalars/arrays (e.g. bai args in the integrator's
    Advanced tree) -- json.dumps(np.float64) raises TypeError, and the
    swallow-all save_session would silently drop the WHOLE write."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def save_session(data: dict, path=None) -> None:
    if path is None and _is_fresh():
        return   # ephemeral fresh session: never write the default file
    p = Path(path) if path else Path(_default_path())
    try:
        with _LOCK:
            # Inline the read so we don't re-enter load_session under the lock.
            p.parent.mkdir(parents=True, exist_ok=True)
            cur = json.loads(p.read_text()) if p.exists() else {}
            cur.update(data)
            p.write_text(json.dumps(cur, indent=2, default=_json_default))
    except Exception:
        logger.debug("save_session failed", exc_info=True)
