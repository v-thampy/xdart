#!/usr/bin/env bash
# xdart one-line installer (macOS / Linux)
#
#   curl -fsSL https://raw.githubusercontent.com/ORG/xrd-tools/main/scripts/install_xdart.sh | bash
#
# Design (docs/design/design_install_and_update_jul2026.md): a pixi WORKSPACE in a
# private app root — conda-forge supplies the compiled I/O stack (the builds
# measured fastest for Eiger bitshuffle/LZ4 decode), PyPI supplies xrd-tools[gui],
# both resolved in ONE solve with a lockfile.  Nothing is assumed on the machine
# (no conda/mamba/python needed); nothing touches an existing conda setup.
# Re-running this script upgrades in place; so does `pixi update` in the app root.
set -euo pipefail

APP_ROOT="${XDART_INSTALL_ROOT:-$HOME/.local/share/xdart}"
BIN_DIR="$HOME/.local/bin"
PYTHON_PIN="3.13.*"     # default interpreter; keep in sync with CI matrix
mkdir -p "$APP_ROOT" "$BIN_DIR"

# --- pixi (self-contained under APP_ROOT; PATH untouched) ---------------------
export PIXI_HOME="$APP_ROOT/pixi"
PIXI="$PIXI_HOME/bin/pixi"
if [ ! -x "$PIXI" ]; then
    echo "==> Installing pixi (self-contained, $PIXI_HOME)"
    curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="$PIXI_HOME" PIXI_NO_PATH_UPDATE=1 bash
fi

# --- workspace manifest --------------------------------------------------------
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  PLAT="osx-arm64" ;;
    Darwin-x86_64) PLAT="osx-64"   ;;
    Linux-x86_64)  PLAT="linux-64" ;;
    Linux-aarch64) PLAT="linux-aarch64" ;;
    *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
# Extras: XDART_EXTRAS is a comma list (default "gui"), e.g.
#   XDART_EXTRAS="gui,fitting"  or  XDART_EXTRAS="all"
# Note: "rsm"/"all" pulls xrayutilities, which has no macOS-arm64 wheels (source
# build needs compilers) — GUI users should stay on the default.
# XDART_LOCAL_SOURCE=/path/to/checkout installs from a local tree (pre-release
# testing / development) instead of PyPI.
EXTRAS_TOML=""
IFS=',' read -ra _extras <<< "${XDART_EXTRAS:-gui}"
for _e in "${_extras[@]}"; do
    _e="${_e// /}"
    [ -n "$_e" ] && EXTRAS_TOML="${EXTRAS_TOML:+$EXTRAS_TOML, }\"$_e\""
done
[ -n "$EXTRAS_TOML" ] || EXTRAS_TOML='"gui"'
GUI_FALLBACK=""
if [ -n "${XDART_LOCAL_SOURCE:-}" ]; then
    PYPI_DEP="xrd-tools = { path = \"$XDART_LOCAL_SOURCE\", extras = [$EXTRAS_TOML] }"
    # Belt-and-suspenders: list the gui stack explicitly for local-source
    # installs (extras on path deps are the least-exercised pixi/uv corner; the
    # duplication is harmless — same versions resolve).  Keep in sync with
    # pyproject [gui].
    case ",${XDART_EXTRAS:-gui}," in *,gui,*)
        GUI_FALLBACK=$'pyside6 = ">=6.5"\npyqtgraph = ">=0.13.7"\nqtawesome = "*"\nimagecodecs = "*"\nimageio = "*"\npackaging = "*"' ;;
    esac
else
    PYPI_DEP="xrd-tools = { version = \">=1,<2\", extras = [$EXTRAS_TOML] }"
fi
cat > "$APP_ROOT/pixi.toml" <<EOF
[workspace]
name = "xdart"
channels = ["conda-forge"]
platforms = ["$PLAT"]

[dependencies]
# conda-forge fast I/O stack (measured: fastest Eiger bitshuffle/LZ4 decode)
python = "$PYTHON_PIN"
h5py = "*"
hdf5plugin = "*"
fabio = "*"
hdf5 = "*"
blosc = "*"
c-blosc2 = "*"
lz4-c = "*"

[pypi-dependencies]
$PYPI_DEP
$GUI_FALLBACK
EOF

echo "==> Solving + installing (conda-forge stack + xrd-tools[gui], one solve)"
"$PIXI" install --manifest-path "$APP_ROOT/pixi.toml"

# Sanity: h5py from the pixi env AND a real LZ4 write/read round-trip through
# hdf5plugin — import alone is not enough; a broken hdf5plugin makes the xdart
# writer silently fall back to UNCOMPRESSED stacks (seen live: "Integrated-stack
# compression = None" → ~4x bigger .nxs + I/O pressure on subsequent runs).
"$PIXI" run --manifest-path "$APP_ROOT/pixi.toml" python - <<'EOF'
import h5py, hdf5plugin, numpy as np, os, tempfile
assert '.pixi' in h5py.__file__, f"h5py not from the pixi env: {h5py.__file__}"
p = tempfile.mktemp(suffix='.h5')
try:
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=np.arange(1000.0), **hdf5plugin.LZ4())
    with h5py.File(p, 'r') as f:
        assert float(f['x'][10]) == 10.0
finally:
    if os.path.exists(p):
        os.remove(p)
print(f"    h5py {h5py.__version__} (HDF5 {h5py.version.hdf5_version}) + hdf5plugin LZ4 round-trip OK")
EOF

# Sanity: the GUI stack must actually import (observed live: a solve can succeed
# with the gui extra silently unapplied -> "installed" but xdart won't launch).
case ",${XDART_EXTRAS:-gui}," in *,gui,*)
    "$PIXI" run --manifest-path "$APP_ROOT/pixi.toml" python -c \
        "import PySide6, pyqtgraph; print('    GUI stack: PySide6', PySide6.__version__, '+ pyqtgraph', pyqtgraph.__version__, 'OK')" ;;
esac

# --- launcher shim + update metadata -------------------------------------------
cat > "$BIN_DIR/xdart" <<EOF
#!/usr/bin/env bash
exec "$PIXI" run --manifest-path "$APP_ROOT/pixi.toml" xdart "\$@"
EOF
chmod +x "$BIN_DIR/xdart"

cat > "$APP_ROOT/install_meta.json" <<EOF
{"flavor": "pixi-workspace", "app_root": "$APP_ROOT",
 "update_cmd": ["$PIXI", "update", "--manifest-path", "$APP_ROOT/pixi.toml"],
 "relaunch_cmd": ["$BIN_DIR/xdart"], "installed_by": "install_xdart.sh v2"}
EOF

echo
echo "xdart installed. Launch with:  xdart"
case ":$PATH:" in
    *":$BIN_DIR:"*)
        # Other xdart entry points on PATH (old pip installs, conda envs) cause
        # two real-world failure modes, both observed live 2026-07-08:
        #  - PATH shadowing: the old one is earlier on PATH and always wins;
        #  - shell hash caching: this SESSION ran the old one before, so zsh/bash
        #    keep launching it even though the new shim is first on PATH.
        # A child process cannot flush the parent shell's hash table, so the
        # best an installer can do is detect the hazard and say exactly what to do.
        OTHERS="$(type -ap xdart 2>/dev/null | grep -vx "$BIN_DIR/xdart" || true)"
        if [ -n "$OTHERS" ]; then
            echo
            echo "NOTE: other xdart installs exist on your PATH:"
            echo "$OTHERS" | sed 's/^/        /'
            echo "      If 'xdart' launches the wrong one:"
            echo "        1. run 'hash -r' (or open a new terminal) — clears the shell's cache;"
            echo "        2. still wrong? that install is earlier on PATH — launch"
            echo "           $BIN_DIR/xdart directly, or 'pip uninstall xrd-tools' in the old env."
        fi ;;
    *) echo "NOTE: add $BIN_DIR to your PATH, e.g.:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;;
esac
echo "Upgrade later: re-run this script, or:  $PIXI update --manifest-path $APP_ROOT/pixi.toml"
