# Phase 5 — `io/metadata.py`: Text/PDI Metadata Readers

**Context**: The xdart GUI currently reads per-image metadata from `.txt` and `.pdi`
sidecar files using functions in `xdart/utils/_utils.py` (lines 333-526). These need
to be extracted into `ssrl_xrd_tools/io/metadata.py` as clean, headless functions that
return dictionaries. The SPEC metadata path already exists in `io/spec.py` and does NOT
need to be duplicated — `io/metadata.py` handles only the txt and pdi formats plus a
unified entry point.

**Important**: Work on the `dev` branch of ssrl_xrd_tools.

---

## Prompt 5A — Implement `io/metadata.py`

Replace the stub at `ssrl_xrd_tools/io/metadata.py` with a full implementation.

### Requirements

1. **Module-level setup**: `from __future__ import annotations`, logging via
   `logging.getLogger(__name__)`, Path imports.

2. **`read_txt_metadata(path: Path | str) -> dict[str, float]`**
   - Reads a `.txt` sidecar metadata file.
   - File format has sections delimited by `# Counters\n...\n` and `# Motors\n...\n`.
     Each section has comma-or-equals-separated `key = value` pairs.
   - Also extracts a timestamp from `time: <datetime> # Temp` pattern, stored as
     `epoch` (float, from `datetime.strptime(..., "%a %b %d %H:%M:%S %Y")`).
   - Returns a flat dict merging counters, motors, and extras (epoch).
   - On parse failure, log a warning and return `{}`.

   Reference implementation (from xdart `_utils.py` lines 497-526):
   ```python
   with open(txt_file, 'r') as f:
       data = f.read()
   counters = re.search('# Counters\n(.*)\n', data).group(1)
   cts = re.split(',|=', counters)
   Counters = {c.split()[0]: float(cs) for c, cs in zip(cts[::2], cts[1::2])}
   motors = re.search('# Motors\n(.*)\n', data).group(1)
   cts = re.split(',|=', motors)
   Motors = {c.split()[0]: float(cs) for c, cs in zip(cts[::2], cts[1::2])}
   Time = (data[data.index('time: ') + 5: data.index('# Temp') - 2]).strip()
   d = datetime.strptime(Time, "%a %b %d %H:%M:%S %Y")
   Extras = {'epoch': time.mktime(d.timetuple())}
   ```

3. **`read_pdi_metadata(path: Path | str) -> dict[str, float]`**
   - Reads a `.pdi` (Pilatus Detector Image) sidecar metadata file.
   - Two format variants (try primary, fall back to secondary):
     - **Primary**: Sections `All Counters;...;;# All Motors` and `All Motors;...;#`
       (newlines replaced with `;`, split on `;|=`).
     - **Fallback**: Section between `# Diffractometer Motor Positions for image;# `
       and `;# Calculated Detector Calibration Parameters for image:`.
       If `2Theta` key exists, also create `TwoTheta` alias.
     - **Last resort**: Return `{'TwoTheta': 0.0, 'Theta': 0.0}` for motors.
   - Extract epoch from last non-empty segment after the final `;`.
   - Returns flat dict merging counters, motors, extras.
   - On parse failure, log a warning and return `{}`.

   Reference implementation (from xdart `_utils.py` lines 456-494):
   ```python
   with open(pdi_file, 'r') as f:
       data = f.read()
   data = data.replace('\n', ';')
   try:
       counters = re.search('All Counters;(.*);;# All Motors', data).group(1)
       cts = re.split(';|=', counters)
       Counters = {c.split()[0]: float(cs) for c, cs in zip(cts[::2], cts[1::2])}
       motors = re.search('All Motors;(.*);#', data).group(1)
       cts = re.split(';|=', motors)
       Motors = {c.split()[0]: float(cs) for c, cs in zip(cts[::2], cts[1::2])}
   except AttributeError:
       # Fallback format
       ss1 = '# Diffractometer Motor Positions for image;# '
       ss2 = ';# Calculated Detector Calibration Parameters for image:'
       try:
           motors = re.search(f'{ss1}(.*){ss2}', data).group(1)
           cts = re.split(';|=', motors)
           Motors = {c.split()[0]: float(cs) for c, cs in zip(cts[::2], cts[1::2])}
           Motors['TwoTheta'] = Motors['2Theta']
       except (AttributeError, KeyError):
           Motors = {'TwoTheta': 0.0, 'Theta': 0.0}
       Counters = {}
   Extras = {}
   if len(data[data.rindex(';') + 1:]) > 0:
       Extras['epoch'] = data[data.rindex(';') + 1:]
   ```

4. **`read_image_metadata(image_path: Path | str, meta_format: str = "txt") -> dict[str, float]`**
   - Unified entry point. `meta_format` is one of `"txt"`, `"pdi"`, or `"spec"`.
   - For `"txt"` and `"pdi"`: locate the sidecar file by trying:
     1. `image_path` with extension replaced (e.g., `image_0001.txt`)
     2. `image_path` with extension appended (e.g., `image_0001.tif.txt`)
   - If no sidecar file found, log debug message and return `{}`.
   - For `"spec"`: import and delegate to `ssrl_xrd_tools.io.spec` functions.
     Extract scan number and image number from the filename using patterns:
     - Scan number: regex `_scan(\d+)_\d+` in the filename stem
     - Image number: last `_(\d+)` before the extension
     - SPEC filename: portion between first `_` and `_scan` in the stem
     - Search for the SPEC file in the image's directory and parent directory.
     Use `silx.io.specfile.SpecFile` to read counters and motors.
     Return flat dict of counters | motors.
   - On any error, log warning and return `{}`.

5. **Helper functions** (private):
   - `_parse_kv_pairs(text: str, delimiters: str = ",|=") -> dict[str, float]`:
     Split text on the delimiter regex, take alternating key/value pairs,
     key = first word of the key string (`.split()[0]`), value = `float(v)`.
     Skip pairs where float conversion fails.
   - `_find_sidecar(image_path: Path, ext: str) -> Path | None`:
     Try `image_path.with_suffix(f".{ext}")` then `Path(str(image_path) + f".{ext}")`.
     Return the first that exists, or None.
   - `_extract_scan_info(image_path: Path) -> tuple[str | None, int | None, int]`:
     Parse SSRL filename conventions to extract (spec_filename, scan_number, image_number).
     Filename pattern: `{prefix}_{specname}_scan{N}_{M}.{ext}` where N=scan, M=image.
     Also handle `b_` prefix (background files).

6. **`__all__`**: Export `read_txt_metadata`, `read_pdi_metadata`, `read_image_metadata`.

7. **Conventions**: Follow `CLAUDE.md` — NumPy docstrings, `Path | str` args,
   `from __future__ import annotations`, logging not print, return `{}` on failure.

---

## Prompt 5B — Tests for `io/metadata.py`

Create `tests/test_metadata.py` with comprehensive tests.

### Requirements

1. **Test fixtures** (use `tmp_path`):
   - `txt_meta_file`: Write a sample `.txt` metadata file with known counters
     (`i0 = 1000.0, i1 = 500.0`), motors (`del = 15.5, eta = 0.2`), and a
     timestamp line (`time: Mon Jan 15 10:30:00 2024 # Temp`).
   - `pdi_meta_file_v1`: Write a primary-format `.pdi` file with
     `All Counters` / `All Motors` sections.
   - `pdi_meta_file_v2`: Write a fallback-format `.pdi` file with
     `# Diffractometer Motor Positions for image` section including `2Theta`.
   - `pdi_meta_file_minimal`: Write a minimal `.pdi` that triggers the last-resort
     fallback (returns `TwoTheta: 0.0, Theta: 0.0`).

2. **Test classes**:

   - `TestReadTxtMetadata`:
     - `test_basic_parsing`: Verify counters, motors, and epoch are all present
       with correct float values.
     - `test_missing_file`: Returns `{}` without raising.
     - `test_malformed_file`: Truncated/garbled content returns `{}`.

   - `TestReadPdiMetadata`:
     - `test_primary_format`: Verify counters and motors from v1 fixture.
     - `test_fallback_format`: Verify motors from v2 fixture, including `TwoTheta` alias.
     - `test_minimal_fallback`: Verify `TwoTheta: 0.0, Theta: 0.0` from minimal fixture.
     - `test_epoch_extraction`: Verify epoch is extracted when present.
     - `test_missing_file`: Returns `{}`.

   - `TestReadImageMetadata`:
     - `test_txt_sidecar_found`: Create an image file + `.txt` sidecar, verify
       `read_image_metadata(image, meta_format="txt")` finds and reads it.
     - `test_pdi_sidecar_found`: Same for `.pdi`.
     - `test_sidecar_not_found`: Returns `{}` when no sidecar exists.
     - `test_appended_extension`: Create sidecar as `image.tif.txt` (appended),
       verify it's found.

   - `TestHelpers`:
     - `test_parse_kv_pairs`: Test `_parse_kv_pairs` with comma-equals and
       semicolon-equals delimiters.
     - `test_parse_kv_pairs_bad_values`: Non-numeric values are skipped.
     - `test_find_sidecar_replaced_ext`: Finds `foo.txt` for `foo.tif`.
     - `test_find_sidecar_appended_ext`: Finds `foo.tif.txt` for `foo.tif`.
     - `test_extract_scan_info`: Test with filename
       `prefix_specname_scan42_0003.tif` → `("specname", 42, 3)`.
     - `test_extract_scan_info_with_b_prefix`: `b_prefix_specname_scan10_0001.edf`
       → `("specname", 10, 1)`.
     - `test_extract_scan_info_no_match`: Non-SSRL filename → `(None, None, ...)`.

3. **No external dependencies**: All tests use synthetic files in `tmp_path`.
   SPEC tests are skipped (the spec path in `read_image_metadata` is a separate
   integration concern — `io/spec.py` already has its own tests).

---

## Prompt 5C — Update `io/__init__.py` and `CLAUDE.md`

1. **`io/__init__.py`**: Add imports from `metadata`:
   ```python
   from ssrl_xrd_tools.io.metadata import (
       read_image_metadata,
       read_pdi_metadata,
       read_txt_metadata,
   )
   ```

2. **`CLAUDE.md`**: In the module map, change `io/metadata.py` from
   `# STUB: txt/pdi/log metadata readers` to
   `# ✅ txt/pdi metadata readers + unified read_image_metadata`.

   In the "Working code" list, add:
   `- io/metadata.py — read_txt_metadata, read_pdi_metadata, read_image_metadata`

   In the "Stubs" list, remove the `io/metadata.py` entry.

   In the refactoring plan section, after item 6 ("SPEC metadata parsing"), add:
   `10. Per-image metadata (txt/pdi sidecar files) → `io/metadata.py` ✅`

---

## Execution Order

Run these in order: **5A → 5B → 5C**. After 5B, run `pytest tests/test_metadata.py -v`
and fix any failures before proceeding to 5C.
