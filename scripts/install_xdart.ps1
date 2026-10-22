# xdart one-line installer (Windows, PowerShell)
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/ORG/xdart/main/scripts/install_xdart.ps1 | iex"
#
# Same design as install_xdart.sh: a pixi workspace in a private app root —
# conda-forge compiled I/O stack and xrayutilities + PyPI xdart[gui] in one solve.
# Nothing assumed on the machine.  Re-run to upgrade.
$ErrorActionPreference = "Stop"

$AppRoot   = if ($env:XDART_INSTALL_ROOT) { $env:XDART_INSTALL_ROOT } else { "$env:LOCALAPPDATA\xdart" }
# Extras: $env:XDART_EXTRAS is a comma list (default "gui"), e.g. "gui,fitting" or "all".
$ExtrasToml = ((($env:XDART_EXTRAS, "gui" -ne $null)[0]).Split(",") |
    ForEach-Object { $_.Trim() } | Where-Object { $_ } |
    ForEach-Object { '"' + $_ + '"' }) -join ", "
if (-not $ExtrasToml) { $ExtrasToml = '"gui"' }
# [rsm]'s xrayutilities is compiled: its wheels bundle a second OpenMP runtime that
# aborts beside conda-forge's.  "gui" and "all" include "rsm": take it, and [rsm]'s
# NumPy pin, from conda-forge.  Keep in sync with pyproject [rsm] and install_xdart.sh.
$XuConda = ""
if ($ExtrasToml -match '"(gui|rsm|all)"') { $XuConda = "numpy = `"==2.5.1`"`nxrayutilities = `"==1.7.12`"" }
$BinDir    = "$AppRoot\bin"
$PythonPin = "3.13.*"
New-Item -ItemType Directory -Force -Path $AppRoot, $BinDir | Out-Null

# --- pixi (self-contained under AppRoot) ---------------------------------------
$env:PIXI_HOME = "$AppRoot\pixi"
$Pixi = "$env:PIXI_HOME\bin\pixi.exe"
if (-not (Test-Path $Pixi)) {
    Write-Host "==> Installing pixi (self-contained, $env:PIXI_HOME)"
    $env:PIXI_NO_PATH_UPDATE = "1"
    Invoke-Expression (Invoke-RestMethod -Uri "https://pixi.sh/install.ps1")
}

# --- workspace manifest ----------------------------------------------------------
# XDART_LOCAL_SOURCE=C:\path\to\checkout installs from a local tree (pre-release
# testing / development) instead of PyPI -- mirrors install_xdart.sh.
# Native Qt provenance is unconditional for both published and local GUI/all installs.
$GuiConda = ""
$GuiFallback = ""
if ($ExtrasToml -match '"(gui|all)"') {
    $GuiConda = 'pyside6 = "==6.11.2"'
    $GuiFallback = "pyqtgraph = `"==0.14.0`"`nqtawesome = `"*`"`nimagecodecs = `"*`"`nimageio = `"*`"`npackaging = `"*`""
}
if ($env:XDART_LOCAL_SOURCE) {
    $SrcPath = ($env:XDART_LOCAL_SOURCE -replace '\\', '/')   # TOML-safe path
    $PypiDep = "xdart = { path = `"$SrcPath`", extras = [$ExtrasToml] }"
} else {
    $PypiDep = "xdart = { version = `">=1,<2`", extras = [$ExtrasToml] }"
}
@"
[workspace]
name = "xdart"
channels = ["conda-forge"]
platforms = ["win-64"]

[dependencies]
# conda-forge fast I/O stack (measured: fastest Eiger bitshuffle/LZ4 decode)
python = "$PythonPin"
h5py = "==3.16.0"
hdf5plugin = "==7.1.0"
fabio = "==2025.10.0"
scipy = "==1.18.1"
numexpr = { version = "==2.14.2", build = "py*" }
silx = "==3.1.3"
hdf5 = "*"
blosc = "*"
c-blosc2 = "*"
lz4-c = "*"
$XuConda
$GuiConda

[pypi-dependencies]
$PypiDep
$GuiFallback
"@ | Set-Content -Encoding UTF8 "$AppRoot\pixi.toml"

Write-Host "==> Solving + installing (conda-forge stack + xdart[gui], one solve)"
& $Pixi install --manifest-path "$AppRoot\pixi.toml"
if ($LASTEXITCODE -ne 0) { throw "pixi install failed" }

# Sanity: h5py from the pixi env AND a real LZ4 write/read round-trip through
# hdf5plugin — a broken hdf5plugin makes the xdart writer silently fall back to
# UNCOMPRESSED stacks (~4x bigger .nexus files).
$SanityPy = @"
import h5py, hdf5plugin, numpy as np, os, tempfile
assert '.pixi' in h5py.__file__, f'h5py not from the pixi env: {h5py.__file__}'
p = tempfile.mktemp(suffix='.h5')
try:
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=np.arange(1000.0), **hdf5plugin.LZ4())
    with h5py.File(p, 'r') as f:
        assert float(f['x'][10]) == 10.0
finally:
    if os.path.exists(p):
        os.remove(p)
print(f'    h5py {h5py.__version__} (HDF5 {h5py.version.hdf5_version}) + hdf5plugin LZ4 round-trip OK')
"@
$SanityFile = "$AppRoot\_sanity_check.py"
$SanityPy | Set-Content -Encoding UTF8 $SanityFile
& $Pixi run --manifest-path "$AppRoot\pixi.toml" python $SanityFile
if ($LASTEXITCODE -ne 0) { throw "h5py/hdf5plugin LZ4 verification failed" }
Remove-Item $SanityFile

# Sanity: the GUI stack must actually import (a solve can succeed with the gui
# extra silently unapplied -> "installed" but xdart won't launch).
if ($ExtrasToml -match '"(gui|all)"') {
    & $Pixi run --manifest-path "$AppRoot\pixi.toml" python -c "import PySide6, pyqtgraph; print('    GUI stack: PySide6', PySide6.__version__, '+ pyqtgraph', pyqtgraph.__version__, 'OK')"
    if ($LASTEXITCODE -ne 0) { throw "GUI stack verification failed (PySide6/pyqtgraph missing)" }
}

# --- launcher shim + PATH + update metadata ---------------------------------------
@"
@echo off
"$Pixi" run --manifest-path "$AppRoot\pixi.toml" xdart %*
"@ | Set-Content -Encoding ASCII "$BinDir\xdart.cmd"

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
    Write-Host "==> Added $BinDir to your user PATH (open a NEW terminal to pick it up)"
}

@"
{"flavor": "pixi-workspace", "app_root": "$($AppRoot -replace '\\','\\\\')",
 "update_cmd": ["$($Pixi -replace '\\','\\\\')", "update", "--manifest-path", "$($AppRoot -replace '\\','\\\\')\\\\pixi.toml"],
 "relaunch_cmd": ["$($BinDir -replace '\\','\\\\')\\\\xdart.cmd"], "installed_by": "install_xdart.ps1 v2"}
"@ | Set-Content -Encoding UTF8 "$AppRoot\install_meta.json"

Write-Host ""
Write-Host "xdart installed. Launch with:  xdart   (from a new terminal)"
Write-Host "Upgrade later: re-run this installer, or:  pixi update --manifest-path $AppRoot\pixi.toml"
