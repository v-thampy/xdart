"""Build a standalone, read-only notebook for selected result files."""

from __future__ import annotations

import json
from pathlib import Path


def _code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def results_notebook_text(
    *, kind: str, paths: tuple[str, ...],
) -> str:
    """Return a clean notebook that reads the exact selected result paths."""

    if kind not in {"nexus", "xye"}:
        raise ValueError("results notebook kind is unsupported")
    if (
        type(paths) is not tuple
        or not paths
        or any(type(path) is not str or not Path(path).is_absolute() for path in paths)
    ):
        raise ValueError("results notebook paths must be nonempty and absolute")
    if kind == "nexus" and len(paths) != 1:
        raise ValueError("a results notebook accepts one processed NeXus file")
    if any(Path(path).suffix.casefold() != f".{kind}" for path in paths):
        raise ValueError("results notebook paths do not match their selected kind")

    cells = [
        {
            "cell_type": "markdown",
            "id": "selected-results",
            "metadata": {},
            "source": (
                "# Analyze Selected Results\n\n"
                "This notebook reads the exact result files selected in XDART. "
                "It does not rerun integration or copy data.\n\n"
                "Open it with `xdart-notebook`, in the xdart Pixi kernel, or in an "
                "environment with `xdart[notebook]`. Matplotlib uses the interactive widget backend "
                "(`%matplotlib widget`, also called `%matplotlib ipympl`) and falls "
                "back to static inline figures when `ipympl` is not installed; "
                "the paths below refer to existing local result files. For a NeXus "
                "result, `read_selected_cake()` reads one 2-D frame on demand.\n"
            ),
        },
        _code(
            "imports",
            "import os\n"
            "from pathlib import Path\n"
            "from IPython import get_ipython\n\n"
            "# Equivalent to %matplotlib widget; XDART_NOTEBOOK_BACKEND=inline suits\n"
            "# headless checks. A kernel without ipympl keeps going with static figures\n"
            "# instead of stopping before the selected paths are defined.\n"
            "_shell = get_ipython()\n"
            "if _shell is not None:\n"
            "    _backend = os.environ.get('XDART_NOTEBOOK_BACKEND', 'widget')\n"
            "    try:\n"
            "        _shell.run_line_magic('matplotlib', _backend)\n"
            "    except Exception as _error:\n"
            "        print(f'Matplotlib backend {_backend!r} is unavailable ({_error}). '\n"
            "              'Using static inline figures; interactive ones need the ipympl '\n"
            "              'package in a Jupyter kernel.')\n"
            "        try:\n"
            "            _shell.run_line_magic('matplotlib', 'inline')\n"
            "        except Exception:\n"
            "            pass\n\n"
            "import matplotlib.pyplot as plt\n"
            "import hdf5plugin  # Register filters for compressed NeXus results.\n"
            "from xrd_tools.io import get_metadata, open_scan, read_xye\n\n"
            f"RESULT_KIND = {kind!r}\n"
            f"RESULT_PATHS = tuple(Path(path) for path in {paths!r})\n"
            "result_paths = RESULT_PATHS"
        ),
        _code(
            "load-results",
            "if RESULT_KIND == 'nexus':\n"
            "    result_path = RESULT_PATHS[0]\n"
            "    scan = open_scan(result_path)\n"
            "    metadata = get_metadata(result_path)\n"
            "    frame_labels = scan.frame_indices\n"
            "    selected_frame = frame_labels[0]\n"
            "    pattern = scan.get_1d(selected_frame)\n"
            "    selected_outputs = {'path': result_path, 'frame_labels': frame_labels}\n"
            "else:\n"
            "    xye_results = []\n"
            "    for path in RESULT_PATHS:\n"
            "        x, intensity, sigma = read_xye(path)\n"
            "        xye_results.append({'path': path, 'x': x, 'intensity': intensity,\n"
            "                            'sigma': sigma})\n"
            "    metadata = {'kind': 'selected_xye', 'count': len(xye_results)}\n"
            "    selected_outputs = {'paths': RESULT_PATHS, 'count': len(xye_results)}\n\n"
            "metadata, selected_outputs"
        ),
        _code(
            "starter-plot",
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "if RESULT_KIND == 'nexus':\n"
            "    ax.plot(pattern.q, pattern.intensity, label=f'frame {selected_frame}')\n"
            "    ax.set_xlabel(pattern.q_unit or 'radial coordinate')\n"
            "else:\n"
            "    for item in xye_results:\n"
            "        ax.plot(item['x'], item['intensity'], label=item['path'].name)\n"
            "    ax.set_xlabel('x')\n"
            "ax.set_ylabel('Intensity')\n"
            "ax.legend()\n"
            "fig.tight_layout()"
        ),
        _code(
            "read-cake",
            "def read_selected_cake(frame_label=None):\n"
            "    \"\"\"Read one 2-D integrated frame on demand; never a whole stack.\"\"\"\n"
            "    if RESULT_KIND != 'nexus':\n"
            "        raise ValueError('XYE selections do not contain a 2-D integrated stack')\n"
            "    return scan.get_2d(selected_frame if frame_label is None else int(frame_label))"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


__all__ = ["results_notebook_text"]
