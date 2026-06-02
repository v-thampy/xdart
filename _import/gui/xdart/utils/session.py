import json
import logging
import threading
from pathlib import Path

# A module-level Path instance is NOT safe to share: Python 3.12's Path
# lazily caches ``_str``/``_drv`` on first use, and that mutation races
# under concurrency (the batch worker + GUI thread both touching it), which
# manifests as a RecursionError deep in pathlib.  Keep the default as a
# plain string and build a fresh Path per call instead.
_DEFAULT = str(Path.home() / '.xdart' / 'session.json')
_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def load_session(path=None) -> dict:
    p = Path(path) if path else Path(_DEFAULT)
    try:
        with _LOCK:
            return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        # Session persistence is a convenience — it must never crash a run.
        logger.debug("load_session failed", exc_info=True)
        return {}


def save_session(data: dict, path=None) -> None:
    p = Path(path) if path else Path(_DEFAULT)
    try:
        with _LOCK:
            # Inline the read so we don't re-enter load_session under the lock.
            p.parent.mkdir(parents=True, exist_ok=True)
            cur = json.loads(p.read_text()) if p.exists() else {}
            cur.update(data)
            p.write_text(json.dumps(cur, indent=2))
    except Exception:
        logger.debug("save_session failed", exc_info=True)
