"""The shared calibration and mask windows (AUD-9 stage 2b G6).

Moved from Stitch's page package so every page that calibrates its own data opens the same window, code and
files (the maintainer, 2026-10-01: "the idea is to reuse what was developed for stitch, not use the
calibration that was done in stitch"). Each page opens its own window with its own
``dialog.CalibrationContext``.
"""
