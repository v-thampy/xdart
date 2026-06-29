# xdart — UI Redesign Spec (Direction A)

A modernization of the xdart static-scan GUI that **keeps every existing control** and changes only hierarchy, grouping, spacing, and readability. Live mockup: `xdart Redesign A.dc.html` (dark/light toggle + mode switcher in the header).

> Repo: `v-thampy/xrd-tools` · Stack: PySide6/PyQt + pyqtgraph · Theme today: a Dracula-derived dark QSS.

---

## 1. Goals

| Pain point (current) | Fix in redesign |
|---|---|
| Cramped twin Scans + Frames lists | Same two lists, but breathing room, real column headers, consistent row padding |
| Weak right-panel hierarchy | Four labeled **cards** (Project & Calibration / Data & Output / Integration / run dock) instead of flat stacked rows |
| Too many tiny buttons, unclear grouping | Integration boxed into 1-D / 2-D sub-cards; range rows aligned on a grid |
| Inconsistent spacing & alignment | One spacing scale (see §3); every field row uses the same label width + gap |
| Paths truncate from the wrong end (`r/LaB6_…`) | Middle/last ellipsis so the **filename stays visible** (your `_TailLineEdit` already does this — apply it everywhere) |
| Clipped labels (`Average S…`, `Write Mod…`) | Labels given room; never clipped |
| Metadata table eats the bottom-left corner | Metadata moves to an on-demand **popup** (button: "Metadata ▾"); the corner becomes a **Tools** slot for planned Fitting / Plot-Metadata |

---

## 2. Layout (unchanged 3-column split)

```
┌ menubar (File · Config) ───────────────────────────────────────────────┐
├ LEFT 296px ──────┬ CENTER (flex) ─────────────────┬ RIGHT 334px ────────┤
│ Data Browser     │ top toolbar (mode-dependent)   │ Calibrate · MakeMask│
│  Scans │ Frames  │ ┌ plot region ───────────────┐ │ ┌ Project & Calib ┐ │
│                  │ │ mode-specific (see §5)     │ │ ┌ Data & Output   ┐ │
│ Show All/AutoLast│ │                            │ │ ┌ Integration     ┐ │
│ Metadata ▾       │ └────────────────────────────┘ │ run dock (status,   │
│ ── Tools ──      │                                │  mode, Batch, Cores,│
│  (planned)       │                                │  Live/Start/Stop)   │
└──────────────────┴────────────────────────────────┴─────────────────────┘
```

Responsive: window is fluid (`max-width:100%`); panels keep fixed side widths, the plot area absorbs the rest. Works laptop → large SLAC monitor.

---

## 3. Design tokens

**Type** — IBM Plex Sans (UI) + IBM Plex Mono (paths, numbers, axis labels). Sizes: section header 11px/700/uppercase, label 11px, body 12px, field text 11–12px, title 13.5px mono.

**Spacing** — 5 / 7 / 8 / 11 / 13–15px. Card radius 8px, field radius 5px, control radius 6px.

**Color (semantic, two themes).** Define these once and swap the set — see §6.

| Token | Dark | Light | Used for |
|---|---|---|---|
| `win-bg` | `#21232e` | `#ffffff` | window body |
| `panel` | `#1e2029` | `#f5f6fa` | side panels |
| `card` | `#191b23` | `#ffffff` | cards, list bg |
| `field` | `#2d3040` | `#eef0f5` | inputs, buttons |
| `field-border` | `#3c4052` | `#d7dbe6` | borders |
| `border` | `#2b2e3b` | `#e1e4ec` | dividers |
| `text` / `text-2` / `text-3` / `muted` | `#eef0f7` / `#cfd4e3` / `#9aa0b5` / `#828799` | `#1b2030` / `#2b3043` / `#5a6075` / `#8a90a2` | text ramp |
| `accent` (+`-soft`,`-text`) | `#bd93f9` | `#7c5cff` | active states, 1-D/2-D tags, Auto/GI |
| `browse` | `#5269a8` | `#3a6fd6` | Browse buttons |
| `start` | `#46c98a` | `#1f9d57` | Start |
| `stop` | `#3a2730`/`#e08597` | `#fff`/`#c0392b` | Stop |
| `plot2d-bg` / `plot1d-bg` | `#0f1118` / `#eceef2` | `#fff` / `#fbfbfd` | plot backgrounds |

(Dark accent kept your existing `#bd93f9`; Browse kept your `#5269a8`.)

---

## 4. Key changes, in order

1. **Right panel → cards.** Each group is a card with an uppercase 11px header: **Project**, **Data**, **Output**, **Integration**, plus the bottom run dock. Replaces the flat collapsible rows.
2. **Wrangler split into three cards: Project / Data / Output.** **Project** holds the Project Folder. **Data** holds everything needed to read frames — Calibration (.poni), Source, Image File, Average series, Meta file, Write mode, Mask file. **Output** is its own card holding the Save Path. (Replaces the old flat Signal + Background groups.)
3. **Integration boxed.** 1-D and 2-D each become a sub-card with: type dropdown, Pts, and two aligned range rows (`q`/`χ`, min → max, Auto). Threshold + Mask-sat row and the Reintegrate/Advanced row sit below.
4. **Paths fixed.** All path fields ellipsize so the **end/filename** is visible (apply `_TailLineEdit` everywhere paths appear: Project, Calibration, Image File, Mask, Save Path).
5. **Metadata popup.** The bottom-left metadata table is removed from the layout; a **Metadata ▾** button opens it as a modal. Frees the corner.
6. **Tools slot.** The freed corner is a labeled, dashed **Tools** placeholder with `PLANNED` chips for **Peak Fitting** and **Plot Metadata** — reserved space for those modules.
7. **Run dock.** Status line + Mode selector + Batch + Cores + Live/Start/Stop grouped at the bottom of the right panel with consistent spacing. Start = `start` green, Stop = muted red.
8. **Odd undo glyphs removed** — the small reset arrows next to each field are dropped (use a single right-click "reset to default" or a per-card reset if needed).

---

## 5. Per-mode behavior (the 4 screens)

The same shell hosts every mode; only the **top toolbar**, **plot region**, and **right-panel enabled-state** change. Mode is set by the bottom-right selector (mirrored in the mockup header).

| Mode | Top toolbar | Plot region | Right panel |
|---|---|---|---|
| **Int 2D** | Norm Channel · Set BG · *title* · Default/Log | Detector 2D + caked Q–χ (top), 1-D plot (bottom) with the 2-D plot toolbar (Q, χ Range, min/max, Single, Options, Legend, Clear, Share Axis, Q-χ) | all enabled |
| **Int 1D** | Norm Channel · Set BG · *title* · **Raw** · Default/Log | Full-height 1-D plot; its toolbar = Q, Single, Options, Clear, **Intensity slider + Autoscale** | 2-D integration block + **Reintegrate 2D** greyed |
| **Image Viewer** | Set BG · *title* · **Intensity slider + Autoscale** · Default/Log | Full-height raw detector image | **Integration card greyed**, run dock greyed (viewing only) |
| **XYE Viewer** | Set BG · *title* · **Intensity slider + Autoscale** · Default/Log | Full-height 1-D scatter of the `.xye` | **Integration card greyed**, run dock greyed |

Notes:
- In the real app, the viewer modes also collapse Calibration/Signal headers to just Project + Save Path. The redesign keeps those cards visible but **greyed** (opacity ~0.4, non-interactive) so the layout doesn't jump. Either is fine — greying is the gentler option.
- Metadata stays a popup in **all** modes (in the viewers it replaces the old inline `source_file` table).

---

## 6. Implementation notes (PySide6 / QSS)

**Theming.** Today the QSS is hard-coded dark. Convert to **two palettes + token substitution**:
- Put every color from §3 into a `dict` per theme (`DARK`, `LIGHT`).
- Keep the QSS as a template string with `{field}`, `{accent}`, … placeholders; `.format(**palette)` on theme change and re-`setStyleSheet()` on the top-level widget.
- Add a Config ▸ Theme toggle (or a small header control). Persist the choice in your settings.

**Files to touch** (from the repo):
- `src/xdart/gui/tabs/static_scan/ui/staticUI.py` — the 3-column splitter + right wrangler stack → regroup into cards / `QGroupBox`es: **Project** (Project Folder), **Data** (Calibration .poni + Source, Image File, Average series, Meta file, Write mode, Mask file), and a separate **Output** group (Save Path). Delete the old Signal/Background groups.
- `src/xdart/gui/tabs/static_scan/ui/integratorUI.py` — wrap 1-D / 2-D into two `QGroupBox` sub-cards; align the range rows on a `QGridLayout` (col widths: tag 44 / min / "to" / max / Auto).
- `src/xdart/gui/tabs/static_scan/ui/h5viewerUI.py` — remove the inline metadata tree from the bottom-left; add a **Metadata** `QPushButton` that opens a `QDialog` populated from the same model. Put a `QFrame` "Tools" placeholder in the vacated space.
- `src/xdart/gui/tabs/static_scan/ui/displayFrameUI.py` — the top toolbar + plot stack; drive the per-mode toolbar/visibility table in §5 off the existing mode enum. Grey integration controls via `setEnabled(False)` per mode.
- `src/xdart/gui/gui_utils.py` — `_TailLineEdit` already exists; use it for **all** path `QLineEdit`s. Centralize the palette dicts + QSS template here.
- `src/xdart/gui/static_controls.py` — run dock grouping (status, mode, Batch, Cores, Live/Start/Stop).

**Don't change:** functional behavior, signal/slot wiring, the integration backend, file IO. This is layout/QSS only.

---

## 7. Claude Code prompt

A ready-to-paste prompt for implementing this in the repo lives in **`CLAUDE_CODE_PROMPT.md`**.
