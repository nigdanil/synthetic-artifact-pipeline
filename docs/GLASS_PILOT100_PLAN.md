# Glass Pilot 100 — step 1: separate private plan (no generation)

**Status:** plan-only implementation. **This step does not generate 100 images.**
Stage-9 `reports/pilot100/` and its 100 already exported bbox-local candidates
must not be modified. Human QA approval and YOLO training eligibility remain pending.

## Objective

Produce **a second, independent series of 100 candidates** whose artifacts are
reflections on the glass lid of a freezer, rather than bright spots inside
individual product bounding boxes. The supplied reference photo contains
broad soft glare and multiple elongated reflections from overhead lamps across
several products. This step prepares their sources and reproducible parameters.

The existing `apply_glass_reflection()` homography-based engine supports light
bands on a manually marked plane. It **does not locate glass automatically**:
the vendor `.seg` labels products/other bbox classes, not glass polygons. A
single quadrilateral copied across all photos would be geometrically invalid.

## Create the plan on Windows (Git Bash, from repository root)

```bash
python scripts/glass_pilot100.py plan
python scripts/glass_pilot100.py check
```

The private `reports/glass_pilot100/plan.json` is created only once; an existing
plan is **never overwritten**. The plan contains 60 new, SHA-distinct learning
sources (not present in existing exported runs), 120 deterministic candidates,
and a goal of **100** technically passing outputs. At least 50 different source
scenes would be required to reach the goal with two variants per scene.

Profiles in each candidate are **initial, uncalibrated proposals**:

| Variant | Band count | Width | Intensity | Target appearance |
|---|---:|---:|---:|---|
| `glass_diffuse` | 1 | 0.13 | 0.50 | broad, diffuse overhead light veil |
| `glass_fluorescent` | 3 | 0.05 | 0.75 | several narrow reflected lamp streaks |

Both currently use `angle_deg=-12` (relative to the local glass plane),
`edge_feather=0.035`, and a fixed per-run seed. The values are **not asserted to
match every freezer**; compare actual renders to the reference photograph
before attempting the full batch. These names are *candidate types*, not new
annotated `glint` objects.

## Required next gate — manually verified glass coordinates

Each `sources[]` record initially sets `glass_quad_norm: null`. Do **not** run a
batch with null coordinates or fill all photos with a default quad.
For each chosen JPEG, the visible glass must be outlined with four normalized
`[x,y]` coordinates in **TL, TR, BR, BL** order and **visually verified**.
This plan validator checks shape and source identity but cannot judge whether
the quadrilateral actually follows that particular freezer's glass.
Only `glass_quad_norm` may be edited in the saved plan. Source hashes, variant
names, presets, run IDs, and seeds are validated and not editable. The next
implementation task is glass-plane review/3–5 real-image smoke tests, **not**
100 unreviewed exports.

## Integrity and limitations

- Original ZIP/SEG, first Pilot 100, Stage-7 bundles and manual reviews: read-only.
- No source overlapping existing exported runs; additionally reject images
  whose SHA-256 occurs more than once in the dataset index.
- Use `learning` only. No `verification` photographs, auto relabeling, or
  automatic `training_approved=true`.
- No ComfyUI/Torch modifications and no upload to GitHub of real images,
  private plan, glass quadrilaterals, manifests or generated results.
- Reserve 20 extra *candidates*, not a guarantee of 100 successful QA samples.
  If unsuitable photos or QA failures exhaust the reserve, stop for review.
- The project still requires a separate effect renderer, reproducible packing,
  strict QA, and human inspection before declaring a second 100-image result.
