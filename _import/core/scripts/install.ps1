<#
.SYNOPSIS
    One-liner installer for ssrl_xrd_tools and xdart on Windows (PowerShell).

.DESCRIPTION
    Native PowerShell port of scripts/install.sh. Avoids the Git Bash
    dependency and reuses whichever conda/mamba is already on PATH in a
    normal Windows shell (cmd.exe, PowerShell, Windows Terminal).

.EXAMPLE
    # Remote (no clone needed) — run from a PowerShell prompt:
    iex "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) }"

.EXAMPLE
    # Remote with options:
    $cmd = "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) } -Name myenv -Force"
    iex $cmd

.EXAMPLE
    # From a local clone:
    .\scripts\install.ps1 -Dev

.PARAMETER Name
    Conda environment name. Default: xrd.

.PARAMETER Python
    Python version. Default: 3.12.

.PARAMETER Dev
    Editable install from the local clone (implies local mode).

.PARAMETER WithXdart
    Use a local xdart clone at this path.

.PARAMETER NoXdart
    Skip the xdart install; install only ssrl_xrd_tools.

.PARAMETER Bootstrap
    Install Miniforge into %USERPROFILE%\miniforge3 if conda is missing.

.PARAMETER Force
    Remove any existing environment with the same name before creating.

.PARAMETER Branch
    Git branch for remote installs. Default: dev.
#>

[CmdletBinding()]
param(
    [string]$Name       = "",
    [string]$Python     = "3.12",
    [switch]$Dev,
    [string]$WithXdart  = "",
    [switch]$NoXdart,
    [switch]$Bootstrap,
    [switch]$Force,
    [string]$Branch     = "dev"
)

$ErrorActionPreference = "Stop"

# ---- config ---------------------------------------------------------------
$GhUser     = "v-thampy"
$SsrlRepo   = "ssrl_xrd_tools"
$XdartRepo  = "xdart"
$DefaultEnv = "xrd"

# ---- local-vs-remote detection -------------------------------------------
# MyInvocation.MyCommand.Path is empty when invoked via
# `iex "& { $(iwr ...) }"` (curl|bash equivalent); in that case we're in
# remote mode. Otherwise treat a sibling environment.yml as the indicator
# of a local clone.
$ScriptPath = $MyInvocation.MyCommand.Path
$LocalMode  = $false
$RepoRoot   = $null
if ($ScriptPath -and (Test-Path $ScriptPath)) {
    $ScriptDir = Split-Path -Parent $ScriptPath
    if (Test-Path (Join-Path $ScriptDir "..\environment.yml")) {
        $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
        $LocalMode = $true
    }
}

if ($Dev -and -not $LocalMode) {
    Write-Error "-Dev requires running from inside a local ssrl_xrd_tools clone."
    exit 1
}

# ---- env name prompt ------------------------------------------------------
$EnvNameExplicit = -not [string]::IsNullOrWhiteSpace($Name)
if (-not $EnvNameExplicit) {
    $prompt = Read-Host -Prompt "Conda environment name [$DefaultEnv]"
    if ([string]::IsNullOrWhiteSpace($prompt)) {
        $Name = $DefaultEnv
    } else {
        $Name = $prompt
    }
}

Write-Host "============================================================"
Write-Host "ssrl_xrd_tools / xdart installer (Windows PowerShell)"
Write-Host "  Env name:       $Name"
Write-Host "  Python:         $Python"
if ($LocalMode) {
    Write-Host "  Source:         local clone at $RepoRoot"
} else {
    Write-Host "  Source:         github.com/$GhUser/$SsrlRepo (branch: $Branch)"
}
Write-Host "  Developer mode: $($Dev.IsPresent)"
Write-Host "============================================================"

# ---- locate or bootstrap conda -------------------------------------------
function Find-CondaExe {
    <#
    .SYNOPSIS
        Return the first conda.exe we can find. Unlike Windows cmd.exe,
        a fresh PowerShell session may not have conda on PATH even when
        it was installed by the standard Anaconda/Miniconda/Miniforge
        installer — those installers set up PATH for cmd.exe sessions
        but require shell-init for PowerShell. We look in the common
        install locations so the user doesn't need to re-install conda
        just to use this script.
    #>
    $candidates = @()

    # 1. Already on PATH?
    $onPath = Get-Command conda -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Path }

    # 2. Common install directories.
    $roots = @(
        (Join-Path $env:USERPROFILE "miniforge3"),
        (Join-Path $env:USERPROFILE "Miniforge3"),
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "Miniconda3"),
        (Join-Path $env:USERPROFILE "anaconda3"),
        (Join-Path $env:USERPROFILE "Anaconda3"),
        (Join-Path $env:USERPROFILE ".conda"),
        "C:\ProgramData\miniforge3",
        "C:\ProgramData\Miniforge3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\Miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\Anaconda3"
    )
    foreach ($root in $roots) {
        $cand = Join-Path $root "Scripts\conda.exe"
        if (Test-Path $cand) { return $cand }
        $cand = Join-Path $root "condabin\conda.bat"
        if (Test-Path $cand) { return $cand }
    }

    # 3. Registry-registered install location (Anaconda/Miniconda)
    try {
        $reg = Get-ItemProperty 'HKLM:\SOFTWARE\Python\ContinuumAnalytics\*' -ErrorAction Stop |
            Select-Object -First 1 -ExpandProperty InstallPath -ErrorAction Stop
        if ($reg -and (Test-Path (Join-Path $reg "Scripts\conda.exe"))) {
            return (Join-Path $reg "Scripts\conda.exe")
        }
    } catch {}

    return $null
}

function Bootstrap-Miniforge {
    $mfDir = Join-Path $env:USERPROFILE "miniforge3"
    if (Test-Path $mfDir) {
        Write-Host "Miniforge already present at $mfDir; skipping bootstrap."
    } else {
        $url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
        $tmp = [System.IO.Path]::GetTempFileName() + ".exe"
        Write-Host "Downloading miniforge installer..."
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $tmp
        Write-Host "Installing miniforge to $mfDir..."
        # /S = silent, /D= (no quotes) = target directory. /D MUST be last.
        $args = "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$mfDir"
        Start-Process -FilePath $tmp -ArgumentList $args -Wait
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Write-Host "Miniforge installed to $mfDir"
    }
    $env:PATH = "$mfDir\Scripts;$mfDir\condabin;$env:PATH"
}

$CondaExe = Find-CondaExe
if (-not $CondaExe) {
    if ($Bootstrap) {
        Bootstrap-Miniforge
        $CondaExe = Find-CondaExe
    }
    if (-not $CondaExe) {
        Write-Error @"
Neither mamba nor conda found.

Either install Miniforge manually (https://github.com/conda-forge/miniforge)
or re-run this script with -Bootstrap to install it automatically.

If you already installed Anaconda/Miniconda but this script didn't pick it
up, you may need to open a fresh PowerShell and run 'conda init powershell'
before re-trying.
"@
        exit 1
    }
}

# Prefer mamba if installed, for speed.
$condaBase = (& $CondaExe info --base).Trim()
$MambaExe = $null
foreach ($cand in @((Join-Path $condaBase "Scripts\mamba.exe"),
                    (Join-Path $condaBase "condabin\mamba.bat"))) {
    if (Test-Path $cand) { $MambaExe = $cand; break }
}
$CondaCmd = if ($MambaExe) { $MambaExe } else { $CondaExe }
Write-Host "Using: $CondaCmd"

# ---- fetch or locate environment.yml -------------------------------------
$TmpDir = Join-Path $env:TEMP ("ssrl_xrd_install_" + [System.Guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
try {
    if ($LocalMode) {
        $EnvFile = Join-Path $RepoRoot "environment.yml"
    } else {
        $EnvFile = Join-Path $TmpDir "environment.yml"
        $EnvUrl = "https://raw.githubusercontent.com/$GhUser/$SsrlRepo/$Branch/environment.yml"
        Write-Host "Fetching environment spec from $EnvUrl"
        Invoke-WebRequest -UseBasicParsing -Uri $EnvUrl -OutFile $EnvFile
    }

    # ---- auto-detect xdart clone in local mode ---------------------------
    if ($LocalMode -and [string]::IsNullOrEmpty($WithXdart) -and -not $NoXdart) {
        $candidates = @(
            (Join-Path (Split-Path -Parent $RepoRoot) "xdart"),
            (Join-Path $env:USERPROFILE "repos\xdart")
        )
        foreach ($cand in $candidates) {
            if ((Test-Path (Join-Path $cand "pyproject.toml")) -or
                (Test-Path (Join-Path $cand "setup.py"))) {
                $WithXdart = $cand
                Write-Host "Auto-detected xdart clone at: $WithXdart"
                break
            }
        }
    }

    # ---- remove existing env if --force ----------------------------------
    $existingEnvs = (& $CondaExe env list) -split "`n" |
        ForEach-Object { ($_ -split "\s+")[0] }
    if ($existingEnvs -contains $Name) {
        if ($Force) {
            Write-Host "Removing existing env '$Name'..."
            & $CondaCmd env remove -n $Name -y
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        } else {
            Write-Error "Environment '$Name' already exists. Use -Force to replace it."
            exit 1
        }
    }

    # ---- patch env spec (name + python version) --------------------------
    $PatchedEnv = Join-Path $TmpDir "environment.patched.yml"
    (Get-Content $EnvFile) `
        -replace '^name:\s.*', "name: $Name" `
        -replace '^  - python=.*', "  - python=$Python" |
        Set-Content $PatchedEnv

    Write-Host ""
    Write-Host "Creating env '$Name' from environment.yml..."
    & $CondaCmd env create --yes --quiet -f $PatchedEnv
    if ($LASTEXITCODE -ne 0) {
        Write-Error @"
'$CondaCmd env create' exited with status $LASTEXITCODE.
The environment may be partially created. Common causes:
  - one of the pip packages in environment.yml failed to build
  - a package conflict that the solver is treating as an error
Re-run with -Force to wipe and retry.
"@
        exit $LASTEXITCODE
    }

    # ---- resolve the env's python interpreter by path --------------------
    $envListOut = & $CondaExe env list
    $envLine = ($envListOut -split "`n" | Where-Object { ($_ -split "\s+")[0] -eq $Name } | Select-Object -First 1)
    if (-not $envLine) {
        Write-Error "env create reported success but '$Name' is not in 'conda env list'."
        exit 1
    }
    $envPrefix = ($envLine -split "\s+")[-1]
    $envPython = Join-Path $envPrefix "python.exe"
    if (-not (Test-Path $envPython)) {
        Write-Error "Could not locate python.exe for env '$Name' (expected at $envPython)"
        exit 1
    }

    # ---- build package list ---------------------------------------------
    $pkgs = @()
    $SsrlGit  = "git+https://github.com/$GhUser/$SsrlRepo.git@$Branch"
    $XdartGit = "git+https://github.com/$GhUser/$XdartRepo.git@$Branch"

    if ($LocalMode -and $Dev) {
        $pkgs += @("-e", $RepoRoot)
    } elseif ($LocalMode) {
        $pkgs += $RepoRoot
    } else {
        $pkgs += $SsrlGit
    }

    if (-not $NoXdart) {
        if ($WithXdart -and $Dev) {
            $pkgs += @("-e", $WithXdart)
        } elseif ($WithXdart) {
            $pkgs += $WithXdart
        } else {
            $pkgs += $XdartGit
        }
    }

    Write-Host ""
    if ($NoXdart) {
        Write-Host "Installing ssrl_xrd_tools into '$Name' (branch: $Branch)..."
    } else {
        Write-Host "Installing xdart and ssrl_xrd_tools into '$Name' (branch: $Branch)..."
    }
    & $envPython -m pip install @pkgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install into '$Name' exited with status $LASTEXITCODE. See the pip output above."
        exit $LASTEXITCODE
    }

    # ---- verify import --------------------------------------------------
    if (-not $NoXdart) {
        & $envPython -c "import xdart" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "xdart was not importable from env '$Name' after install."
            exit 1
        }
        Write-Host "Verified: xdart is importable from '$Name'."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Environment '$Name' is ready."
    Write-Host ""
    Write-Host "To activate in a new PowerShell:"
    Write-Host "  conda activate $Name"
    Write-Host ""
    Write-Host "If 'conda activate' fails in PowerShell, run 'conda init powershell'"
    Write-Host "once (as your user) and open a fresh PowerShell."
    if (-not $NoXdart) {
        Write-Host "  xdart         # launch the xdart GUI"
    }
    if ($Dev) {
        Write-Host ""
        Write-Host "Dev mode: edits to the local repo(s) take effect immediately."
    } else {
        Write-Host ""
        Write-Host "To update later, re-run this installer with -Force."
    }
    Write-Host "============================================================"
}
finally {
    if (Test-Path $TmpDir) {
        Remove-Item $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
