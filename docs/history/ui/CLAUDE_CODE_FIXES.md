# xdart redesign — CORRECTIVE prompt for Claude Code (round 2)

The first pass got the structure right but missed the visual styling because the instructions
were too abstract for Qt. The biggest issues: sections render as **flat bars** instead of
**rounded cards**, the wrangler grouping is wrong, the Refresh button is misplaced, and some
button labels are clipped. Paste the block below to fix them.

**Use `xdart_reference_all_modes_dark.html` and `xdart_reference_all_modes_light.html` as the visual
targets** (dark + light). They are flat, static HTML mirrors of the mockup (all modes) with
all template logic / JS removed — open them and read exact values straight from the inline styles
(paddings, border-radius, gaps, font sizes, and the color tokens in each file's `:root` block).
Much easier to read than the `.dc.html`.

---

```
Several visual details from the redesign spec didn't come through. Fix these — styling/layout
only, no behavior changes. Visual target: xdart_reference_all_modes_dark.html (a flat static HTML
mirror — read exact paddings, radii, gaps, font sizes, and color tokens directly from it).

1. ROUNDED CARDS (most important). The right-panel sections currently render as flat bars with
   collapse triangles. They must be rounded, contained cards. In Qt, QGroupBox does NOT round
   reliably (its native title conflicts with border-radius). Instead, build each card as a QFrame:

     card = QFrame(); card.setObjectName("card")
     # inside: a header QLabel (objectName "cardHeader") + the field rows

   and style via QSS:
     QFrame#card {
         background: <card>;            /* token: card  (dark #191b23 / light #ffffff) */
         border: 1px solid <border>;    /* token: border */
         border-radius: 8px;
     }
     QLabel#cardHeader {
         font: 700 11px "Source Sans 3";
         letter-spacing: 1px;
         color: <text-3>;
         padding: 9px 12px 6px 12px;    /* header sits inside the card */
         text-transform: uppercase;     /* or upper-case the string yourself */
     }
   Put ~12px padding inside each card and 11px vertical gap BETWEEN cards. Do not use a collapse
   triangle — these are static cards, not collapsibles.

2. WRANGLER GROUPING — three cards, in this exact order:
     • "PROJECT"  → one row: Folder  [path field] [Browse]
     • "DATA"     → rows: Calibration [.poni] [Browse], Source [dropdown], Image File [..][Browse],
                    Average Scan [toggle], Meta File [txt], Write Mode [dropdown], Mask File [..][Browse]
       (Calibration belongs INSIDE Data, at the top — there is NO separate Calibration card.)
     • "OUTPUT"   → one row: Save Path [path field] [Browse]
   Remove the standalone CALIBRATION card and rename BACKGROUND → OUTPUT.

3. CONSISTENT FIELD ROWS. Every parameter row is: [label, fixed ~78px wide][field stretches][Browse if a path].
   Apply this to the Project "Folder" row too (label "Folder", not "Project Folder"). Labels must
   never clip ("Average Scan" etc. need room).

4. DATA BROWSER HEADER + REFRESH. Above the Scans/Frames lists add a header row: a "Data Browser"
   QLabel on the left and the "Refresh" button on the RIGHT of that same row. The bottom button row
   under the lists should then be just: [Show All] [Auto Last] [Metadata ▾]. Move Refresh out of it.

5. BUTTON LABEL CLIPPING. Buttons are showing "Auto Las" and "Ietadata" — they have fixed widths too
   small for their text. Let buttons size to their content (or set a min-width that fits the label
   plus padding). Affected: Auto Last, Metadata. Check Calibrate / Make Mask / Reintegrate too.

6. INTEGRATION SUB-CARDS. Wrap the 1-D and 2-D blocks each in their own rounded inner panel
   (QFrame, background = panel token, border 1px border token, border-radius 6px, ~9px padding),
   matching the outer card treatment one level down. The 1-D / 2-D tags stay as small accent pills.

7. TOOLS PLACEHOLDER. The Tools box should be a dashed rounded container
   (QFrame: border: 1px dashed <field-border>; border-radius: 7px; background: <card>; padding: 14px)
   with its planned-tool rows inside, matching the mockup.

8. PARAMETER LABELS ARE PLAIN TEXT — NOT BOXED. Right now many static labels have a full field
   box (border + background) drawn around them. A label that only NAMES a parameter and isn't
   clickable must be a bare QLabel: no border, no background, no border-radius. Confirmed offenders:
     • both "Pts" labels in the integrator (1-D and 2-D rows)
     • "Motor" in the GI (Fiber) row
     • the "to" separators between min/max fields, "Cores", "Intensity", "Mask sat."
     • the per-row name labels in the wrangler (Source, Image File, Meta File, Write Mode, etc.)
   Only genuine CONTROLS keep a box: text inputs, dropdowns, and buttons.
   Nuance — the radial/azimuthal UNIT selectors ("Q (Å⁻¹)", "χ (°)", "IP (Å⁻¹)", "OOP (Å⁻¹)") ARE
   dropdowns the user can change, so they keep a box — but style them as dropdowns (with a ▾ caret),
   not as plain bordered labels. If any of those is actually fixed/non-selectable, make it plain text.
   Rule of thumb: if clicking it does nothing, it has no box.

Keep everything else (colors, the integration controls, run dock, per-mode behavior) as-is.
After the change, run the app so I can compare side by side with the mockup.
```

---

## Why the first pass missed these (worth knowing)

- **`QGroupBox` ≠ rounded card in Qt.** Its frame + native title don't honor `border-radius` cleanly, so you get flat bars. `QFrame#card` + a `QLabel` header is the reliable pattern. (This is the single biggest reason your output looks different.)
- **Fixed button widths** are common in hand-built Qt layouts → label clipping. Prefer content-sized buttons or a min-width computed from the font metrics.
- **Inter-card spacing** comes from the layout's `setSpacing(11)` + `setContentsMargins`, not from the QSS — set both.
- **Don't put borders on QLabels.** A blanket `QLabel { border: 1px solid ... }` (or styling labels with the same class as fields) is what boxes the parameter names. Scope field styling to the input/dropdown/button classes only.

---

## Fonts

The mockup now uses **Source Sans 3** (UI) + **JetBrains Mono** (paths, numbers, axis labels) —
chosen for legibility in dense numeric fields. To swap the whole app, change just those two families
in the QSS template. Load them from Google Fonts (or bundle the TTFs). Alternatives considered:

1. **IBM Plex Sans + IBM Plex Mono** (current) — one superfamily, technical and neutral.
2. **Source Sans 3 + JetBrains Mono** — SELECTED. Very legible neutral sans; JetBrains Mono has the
   clearest digit/letter distinction (great for all the numeric fields and axis labels).
3. **Public Sans + Space Mono** — utilitarian government-grade sans; Space Mono adds a little
   character in the numbers (use Space Mono only if you like the quirk).
4. **Atkinson Hyperlegible + IBM Plex Mono** — maximum legibility / lowest eye-fatigue for long
   beamline sessions; distinctive but still professional.
5. **Libre Franklin + Roboto Mono** — classic grotesque warmth + a safe, ubiquitous mono.

All are free (Google Fonts / SIL). Avoid leading with Inter/Roboto Sans for the UI — they read as
generic. Whatever you pick, keep mono strictly for paths, numeric inputs, and axis tick labels.
