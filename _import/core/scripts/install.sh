#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install.sh — one-line installer for ssrl_xrd_tools and xdart.
#
# Designed to work both as a remote one-liner and from a local clone:
#
#   # Remote (no clone needed):
#   curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
#
#   # Remote with options:
#   curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- -n myenv --force
#
#   # From a local clone (developer mode, editable installs):
#   ./scripts/install.sh --dev
#
# Options:
#   -n, --name NAME       Conda environment name (default: xrd)
#   -p, --python VERSION  Python version (default: 3.12)
#   --dev                 Editable install from a local clone (implies local mode).
#   --with-xdart PATH     Use a local xdart clone at PATH (editable if --dev)
#   --no-xdart            Skip xdart install
#   --bootstrap           Install miniforge into $HOME/miniforge3 if conda is missing
#   --force               Remove any existing env with the same name first
#   --branch BRANCH       Git branch for remote installs (default: dev)
#   -h, --help            Show this help
#
# Defaults install ssrl_xrd_tools AND xdart from the `dev` branch of
# https://github.com/v-thampy on conda-forge + pip.
# ---------------------------------------------------------------------------

set -euo pipefail

# ----- config --------------------------------------------------------------
GH_USER="v-thampy"
SSRL_REPO="ssrl_xrd_tools"
XDART_REPO="xdart"
DEFAULT_BRANCH="dev"

ENV_NAME="xrd"
PYTHON_VERSION="3.12"
DEV_MODE=0
FORCE=0
BOOTSTRAP=0
WITH_XDART=""
NO_XDART=0
BRANCH="${DEFAULT_BRANCH}"

# Detect if we're running from a local clone (i.e. scripts/install.sh inside
# the ssrl_xrd_tools repo) vs remote via curl|bash. In curl|bash mode,
# BASH_SOURCE[0] is typically empty or /dev/fd/*.
SCRIPT_DIR=""
REPO_ROOT=""
LOCAL_MODE=0
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${SCRIPT_DIR}/../environment.yml" ]]; then
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
        LOCAL_MODE=1
    fi
fi

print_help() {
    sed -n '2,32p' "${BASH_SOURCE[0]:-$0}" 2>/dev/null | sed 's/^# \{0,1\}//' || {
        echo "install.sh — see header comments in the script for usage"
    }
    exit 0
}

# ----- arg parsing ---------------------------------------------------------
if [[ $# -gt 0 && "$1" != -* ]]; then
    ENV_NAME="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)       ENV_NAME="$2"; shift 2 ;;
        -p|--python)     PYTHON_VERSION="$2"; shift 2 ;;
        --dev)           DEV_MODE=1; shift ;;
        --with-xdart)    WITH_XDART="$2"; shift 2 ;;
        --no-xdart)      NO_XDART=1; shift ;;
        --bootstrap)     BOOTSTRAP=1; shift ;;
        --force)         FORCE=1; shift ;;
        --branch)        BRANCH="$2"; shift 2 ;;
        -h|--help)       print_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ${DEV_MODE} -eq 1 && ${LOCAL_MODE} -eq 0 ]]; then
    echo "ERROR: --dev requires running from inside a local ssrl_xrd_tools clone." >&2
    echo "       Clone the repo first, then run ./scripts/install.sh --dev" >&2
    exit 1
fi

echo "============================================================"
echo "ssrl_xrd_tools / xdart installer"
echo "  Env name:       ${ENV_NAME}"
echo "  Python:         ${PYTHON_VERSION}"
if [[ ${LOCAL_MODE} -eq 1 ]]; then
    echo "  Source:         local clone at ${REPO_ROOT}"
else
    echo "  Source:         github.com/${GH_USER}/${SSRL_REPO} (branch: ${BRANCH})"
fi
echo "  Developer mode: $([[ ${DEV_MODE} -eq 1 ]] && echo yes || echo no)"
echo "============================================================"

# ----- miniforge bootstrap -------------------------------------------------
bootstrap_miniforge() {
    local mf_dir="${HOME}/miniforge3"
    if [[ -d "${mf_dir}" ]]; then
        echo "Miniforge already present at ${mf_dir}; skipping bootstrap."
    else
        local os arch url
        os="$(uname -s)"
        arch="$(uname -m)"
        case "${os}-${arch}" in
            Darwin-arm64)   url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" ;;
            Darwin-x86_64)  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
            Linux-x86_64)   url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" ;;
            Linux-aarch64)  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" ;;
            *) echo "ERROR: unsupported platform ${os}-${arch} for bootstrap" >&2; exit 1 ;;
        esac
        echo "Downloading miniforge installer for ${os}-${arch}..."
        local tmp_installer
        tmp_installer="$(mktemp -t miniforge.XXXXXX.sh)"
        curl -fsSL "${url}" -o "${tmp_installer}"
        bash "${tmp_installer}" -b -p "${mf_dir}"
        rm -f "${tmp_installer}"
        echo "Miniforge installed to ${mf_dir}"
    fi
    # Add to PATH for the rest of this script
    export PATH="${mf_dir}/bin:${PATH}"
    # shellcheck disable=SC1091
    source "${mf_dir}/etc/profile.d/conda.sh"
}

# ----- pick mamba or conda -------------------------------------------------
if ! command -v mamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
    if [[ ${BOOTSTRAP} -eq 1 ]]; then
        bootstrap_miniforge
    else
        echo ""
        echo "ERROR: neither mamba nor conda found on PATH." >&2
        echo "Either install miniforge manually (https://github.com/conda-forge/miniforge)" >&2
        echo "or re-run this script with --bootstrap to install it automatically:" >&2
        echo "  curl -sSL ... | bash -s -- --bootstrap" >&2
        exit 1
    fi
fi

if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using: ${CONDA_CMD}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ----- fetch or locate environment.yml -------------------------------------
TMP_DIR="$(mktemp -d -t ssrl_xrd_install.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ ${LOCAL_MODE} -eq 1 ]]; then
    ENV_FILE="${REPO_ROOT}/environment.yml"
else
    ENV_FILE="${TMP_DIR}/environment.yml"
    ENV_URL="https://raw.githubusercontent.com/${GH_USER}/${SSRL_REPO}/${BRANCH}/environment.yml"
    echo "Fetching environment spec from ${ENV_URL}"
    curl -fsSL "${ENV_URL}" -o "${ENV_FILE}"
fi

# ----- auto-detect xdart clone (local mode only) ---------------------------
if [[ ${LOCAL_MODE} -eq 1 && -z "${WITH_XDART}" && ${NO_XDART} -eq 0 ]]; then
    for candidate in \
        "$(dirname "${REPO_ROOT}")/xdart" \
        "${HOME}/repos/xdart"
    do
        if [[ -f "${candidate}/pyproject.toml" || -f "${candidate}/setup.py" ]]; then
            WITH_XDART="${candidate}"
            echo "Auto-detected xdart clone at: ${WITH_XDART}"
            break
        fi
    done
fi

# ----- remove existing env if --force --------------------------------------
if ${CONDA_CMD} env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ ${FORCE} -eq 1 ]]; then
        echo "Removing existing env '${ENV_NAME}'..."
        ${CONDA_CMD} env remove -n "${ENV_NAME}" -y
    else
        echo "ERROR: environment '${ENV_NAME}' already exists. Use --force to replace it." >&2
        exit 1
    fi
fi

# ----- create env ----------------------------------------------------------
PATCHED_ENV="${TMP_DIR}/environment.patched.yml"
sed -e "s/^name: .*/name: ${ENV_NAME}/" \
    -e "s/^  - python=.*/  - python=${PYTHON_VERSION}/" \
    "${ENV_FILE}" > "${PATCHED_ENV}"

echo ""
echo "Creating env '${ENV_NAME}' from environment.yml..."
${CONDA_CMD} env create -f "${PATCHED_ENV}"

conda activate "${ENV_NAME}"

# ----- install xdart (+ ssrl_xrd_tools) ------------------------------------
# xdart declares ssrl_xrd_tools as a dependency, so installing xdart is
# sufficient. We explicitly install ssrl_xrd_tools from the same branch first
# so pip uses that version rather than whatever is on PyPI.
SSRL_GIT_URL="git+https://github.com/${GH_USER}/${SSRL_REPO}.git@${BRANCH}"
XDART_GIT_URL="git+https://github.com/${GH_USER}/${XDART_REPO}.git@${BRANCH}"

PKGS=()
if [[ ${LOCAL_MODE} -eq 1 && ${DEV_MODE} -eq 1 ]]; then
    PKGS+=(-e "${REPO_ROOT}")
elif [[ ${LOCAL_MODE} -eq 1 ]]; then
    PKGS+=("${REPO_ROOT}")
else
    PKGS+=("${SSRL_GIT_URL}")
fi

if [[ ${NO_XDART} -eq 0 ]]; then
    if [[ -n "${WITH_XDART}" && ${DEV_MODE} -eq 1 ]]; then
        PKGS+=(-e "${WITH_XDART}")
    elif [[ -n "${WITH_XDART}" ]]; then
        PKGS+=("${WITH_XDART}")
    else
        PKGS+=("${XDART_GIT_URL}")
    fi
fi

echo ""
if [[ ${NO_XDART} -eq 0 ]]; then
    echo "Installing xdart and ssrl_xrd_tools (branch: ${BRANCH})..."
else
    echo "Installing ssrl_xrd_tools (branch: ${BRANCH})..."
fi
pip install "${PKGS[@]}"

# ----- done ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "Environment '${ENV_NAME}' is ready."
echo ""
echo "To start using it:"
echo "  conda activate ${ENV_NAME}"
if [[ ${NO_XDART} -eq 0 ]]; then
    echo "  xdart         # launch the xdart GUI"
fi
echo ""
if [[ ${DEV_MODE} -eq 1 ]]; then
    echo "Dev mode: edits to the local repo(s) take effect immediately."
else
    echo "To update later, re-run this installer with --force."
fi
echo "============================================================"
