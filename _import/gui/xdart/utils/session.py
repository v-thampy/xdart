import json
from pathlib import Path

_DEFAULT_PATH = Path.home() / '.xdart' / 'session.json'


def load_session(path: Path = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_session(data: dict, path: Path = None) -> None:
    p = Path(path) if path else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    current = load_session(p)
    current.update(data)
    p.write_text(json.dumps(current, indent=2))
