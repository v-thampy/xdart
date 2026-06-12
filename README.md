# xrd-tools

SSRL X-ray diffraction toolkit: a headless reduction core (`xrd_tools`)
and a Qt GUI (`xdart`) in one distribution.

```bash
pip install xrd-tools          # headless core
pip install "xrd-tools[gui]"   # + the xdart GUI
```

```python
import xrd_tools   # headless: io / reduction / core / viz
import xdart       # Qt GUI (requires the gui extra)
```

Full docs, migration notes (`MIGRATION.md`) and development guide to
follow — this repo combines the former `ssrl_xrd_tools` and `xdart`
repositories with their full histories.
