# Publishing to PyPI and conda-forge

## Prerequisites

Install the build and upload tools:

```bash
pip install build twine
```

Create accounts on:
- **PyPI**: https://pypi.org/account/register/
- **Test PyPI**: https://test.pypi.org/account/register/ (use this first!)

Set up API tokens (recommended over username/password):
1. Go to https://pypi.org/manage/account/token/ (or test.pypi.org equivalent)
2. Create a token scoped to your project (or account-wide for first upload)
3. Save the token — you'll need it for `twine upload`

You can store tokens in `~/.pypirc` to avoid entering them each time:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

---

## Step 1: Publish ssrl_xrd_tools first

Since xdart depends on `ssrl_xrd_tools>=0.2.0`, you must publish ssrl_xrd_tools first.

```bash
cd ~/repos/ssrl_xrd_tools
```

### 1a. Verify the package builds cleanly

```bash
# Clean any old builds
rm -rf dist/ build/ *.egg-info

# Build sdist and wheel
python -m build
```

This creates two files in `dist/`:
- `ssrl_xrd_tools-0.2.0.tar.gz` (source distribution)
- `ssrl_xrd_tools-0.2.0-py3-none-any.whl` (wheel)

### 1b. Test the package locally

```bash
# Install in a fresh environment to verify
python -m venv /tmp/test-ssrl
source /tmp/test-ssrl/bin/activate
pip install dist/ssrl_xrd_tools-0.2.0-py3-none-any.whl
python -c "import ssrl_xrd_tools; print(ssrl_xrd_tools.__version__)"
deactivate
rm -rf /tmp/test-ssrl
```

### 1c. Upload to Test PyPI first

```bash
twine upload --repository testpypi dist/*
```

Verify at https://test.pypi.org/project/ssrl_xrd_tools/

Test installing from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ssrl_xrd_tools
```

(The `--extra-index-url` lets pip find dependencies like numpy/scipy from real PyPI.)

### 1d. Upload to real PyPI

```bash
twine upload dist/*
```

Verify at https://pypi.org/project/ssrl_xrd_tools/

---

## Step 2: Publish xdart

```bash
cd ~/repos/xdart
```

### 2a. Build

```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

### 2b. Test locally

```bash
python -m venv /tmp/test-xdart
source /tmp/test-xdart/bin/activate
pip install dist/xdart-0.15.0-py3-none-any.whl
python -c "import xdart; print('OK')"
xdart  # test the console entry point
deactivate
rm -rf /tmp/test-xdart
```

### 2c. Upload to Test PyPI, then real PyPI

```bash
# Test PyPI first
twine upload --repository testpypi dist/*

# Then real PyPI
twine upload dist/*
```

---

## Step 3: conda-forge

conda-forge uses "staged-recipes" — you submit a recipe PR and their CI builds the package.

### 3a. Create the recipe for ssrl_xrd_tools

Fork https://github.com/conda-forge/staged-recipes and clone it:

```bash
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
git checkout -b ssrl_xrd_tools
```

Create `recipes/ssrl_xrd_tools/meta.yaml`:

```yaml
{% set name = "ssrl_xrd_tools" %}
{% set version = "0.2.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
  # Get sha256 from: pip hash dist/ssrl_xrd_tools-0.2.0.tar.gz
  sha256: REPLACE_WITH_ACTUAL_SHA256

build:
  noarch: python
  number: 0
  script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation

requirements:
  host:
    - python >=3.10
    - pip
    - setuptools >=64
  run:
    - python >=3.10
    - numpy
    - scipy
    - h5py
    - fabio
    - silx
    - pyfai
    - xrayutilities
    - joblib
    - natsort
    - lmfit

test:
  imports:
    - ssrl_xrd_tools
    - ssrl_xrd_tools.core.containers
    - ssrl_xrd_tools.integrate.gid
    - ssrl_xrd_tools.io

about:
  home: https://github.com/v-thampy/ssrl_xrd_tools
  license: BSD-3-Clause
  license_family: BSD
  license_file: LICENSE
  summary: SSRL X-ray diffraction data processing and visualization tools
  dev_url: https://github.com/v-thampy/ssrl_xrd_tools

extra:
  recipe-maintainers:
    - v-thampy
```

To get the SHA256 hash:

```bash
pip hash dist/ssrl_xrd_tools-0.2.0.tar.gz
# Or after uploading to PyPI:
curl -s https://pypi.org/pypi/ssrl_xrd_tools/0.2.0/json | python -c "import sys,json; d=json.load(sys.stdin); print([f['digests']['sha256'] for f in d['urls'] if f['filename'].endswith('.tar.gz')][0])"
```

### 3b. Create the recipe for xdart

Create `recipes/xdart/meta.yaml`:

```yaml
{% set name = "xdart" %}
{% set version = "0.15.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
  sha256: REPLACE_WITH_ACTUAL_SHA256

build:
  noarch: python
  number: 0
  script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation
  entry_points:
    - xdart = xdart.xdart_main:main

requirements:
  host:
    - python >=3.10
    - pip
    - setuptools >=64
  run:
    - python >=3.10
    - ssrl_xrd_tools >=0.2.0
    - numpy
    - scipy
    - h5py
    - hdf5plugin
    - fabio
    - silx
    - pyfai
    - xrayutilities
    - pyside6 >=6.5
    - pyqtgraph >=0.13.7
    - pandas
    - lmfit
    - pyyaml
    - matplotlib
    - joblib
    - natsort
    - imagecodecs

test:
  imports:
    - xdart
    - xdart.modules.ewald.arch
  commands:
    - xdart --help || true

about:
  home: https://github.com/v-thampy/xdart
  license: MIT
  license_family: MIT
  license_file: LICENSE
  summary: A pyFAI-based GUI for X-ray diffraction data reduction and visualization
  dev_url: https://github.com/v-thampy/xdart

extra:
  recipe-maintainers:
    - v-thampy
```

### 3c. Submit the PR

```bash
git add recipes/ssrl_xrd_tools/meta.yaml
git commit -m "Add ssrl_xrd_tools recipe"
git push origin ssrl_xrd_tools
```

Open a PR at https://github.com/conda-forge/staged-recipes/pulls. The conda-forge CI will build and test your package. A reviewer will merge it once tests pass.

**Important**: Submit ssrl_xrd_tools first. Once it's on conda-forge, submit xdart in a separate PR (since it depends on ssrl_xrd_tools being available).

### 3d. After acceptance

Once merged, conda-forge creates a "feedstock" repo (e.g., `ssrl_xrd_tools-feedstock`) where you can update future versions. To release a new version, either update the feedstock's `recipe/meta.yaml` or let the conda-forge bot auto-detect new PyPI releases.

---

## Version bumping checklist

When releasing a new version:

1. Update version in `pyproject.toml`
2. Update version in `__init__.py` (ssrl_xrd_tools)
3. Commit and tag: `git tag v0.2.1 && git push --tags`
4. Build: `python -m build`
5. Upload: `twine upload dist/*`
6. conda-forge bot will usually auto-detect the new PyPI version and open a PR on the feedstock

---

## Troubleshooting

**"File already exists" on PyPI**: You cannot overwrite an existing version. Bump the version number.

**Missing dependencies on conda-forge**: All runtime deps must be available on conda-forge. Check https://anaconda.org/conda-forge/PACKAGE_NAME for each dependency.

**PySide6 on conda-forge**: PySide6 may not be available on all platforms via conda-forge. Users may need to install it via pip even in a conda environment: `pip install PySide6`.
