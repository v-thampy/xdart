# Perf baseline — Eiger 651-frame (post conda-forge decode fix) — 2026-06-15

Reference numbers for future comparison. To compare: re-run the same scan in each mode and diff the
`[PERF-SUMMARY]` line.

## Conditions
- **Scan:** `eiger_S069Ta_redo_eta2p0_1_scan001` — 651 frames, Eiger `_master.h5`.
- **Build/env:** `xrd-tools` monorepo in conda env `xrd_test`, WITH the conda-forge native decode stack
  installed (`conda install -c conda-forge h5py hdf5plugin fabio hdf5 blosc c-blosc2 lz4-c`). This is the
  fast-decode baseline — pure-PyPI-wheel decode was ~24 ms/frame vs ~15 ms/frame here.
- **Read path:** fabio per-frame decode (`prefetch_io=0` everywhere; `XDART_EIGER_H5_BULK` OFF;
  `XDART_PREFETCH_QUEUE_SIZE=4` default).
- **Machine:** Mac (LVMHG7CN49). **Date:** 2026-06-15.

## Numbers (651 frames)

| Mode          | Total   | collect_read (queue-wait) | dispatch (reduce+write) | read ms/fr | dispatch ms/fr | save flush     |
|---------------|---------|---------------------------|-------------------------|------------|----------------|----------------|
| Int 1D (Live) | 18.88 s | 9.83 s                    | 8.64 s                  | 15.1       | 13.3           | —              |
| Int 2D (Live) | 24.79 s | 2.12 s                    | 22.19 s                 | 3.3        | 34.1           | —              |
| Int 1D Batch  | 28.34 s | 13.78 s                   | 13.86 s                 | 21.2       | 21.3           | 256 / 256 / 139|
| Int 2D Batch  | 36.29 s | 12.48 s                   | 22.87 s                 | 19.2       | 35.1           | 64×10 / 11     |
| Int 1D XYE    | 26.73 s | 13.54 s                   | 12.72 s                 | 20.8       | 19.5           | 256 / 256 / 139|

## Observations (context for future comparison; not action items)
- **1D Live** is read-bound (collect_read 9.8 > dispatch 8.6) but the read is now fast (~15 ms/fr after
  the conda-forge fix). **2D Live** is dispatch-bound (cake reduction ~34 ms/fr; reads hidden, 3.3 ms/fr
  queue-wait).
- **Batch is notably slower than Live** for the same mode (1D: 28.3 vs 18.9 s; 2D: 36.3 vs 24.8 s), and
  batch's read ms/fr is higher (21 vs 15 for 1D; 19 vs 3 for 2D) — i.e. the batch path overlaps reads
  with reduce less well than live. Worth a look if batch throughput matters (independent of the
  decode-library work).
- **2D Batch flushes every 64 frames** vs 256 for 1D Batch / XYE — smaller save interval for the larger
  2D payload.
- All runs `prefetch_io=0` (fabio per-frame path). The single-thread decode ceiling (~16 s for 1D) and
  the read‖reduce overlap behaviour are documented in the take-stock memory (rounds 15–21).
