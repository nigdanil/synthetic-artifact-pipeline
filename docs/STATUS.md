# Status of the public scaffold (2026-10-13)

The repository contains a tested local image-augmentation prototype (Tasks 1–5), plus a read-only ComfyUI preflight and loader/ROI selector classic custom nodes. The Dataset Loader and ROI Selector were verified in the user's ComfyUI. Mask Builder was confirmed in the user’s ComfyUI (original, bbox overlay, MASK and cyan mask overlay).

**Implemented (Task 1):** read-only ZIP indexer, JPEG/SEG pairing, raw SHA-256 hashes, exact-duplicate/cross-split grouping and private JSON reports. See [dataset indexing](DATASET_INDEXING.md). Confirmed locally on Windows: both archives indexed; Python environment/tests verified.

**Implemented (Task 2):** canonical `.seg` / YOLO TXT bbox parsers, strict geometry checks, explicit class mapping, lossless field preservation on SEG rewrite, and local-only bbox previews. See [annotations](ANNOTATIONS.md). Verify on the user's Windows environment before accepting Task 2.

**Implemented (Task 3):** local ZIP image/SEG loader verifies content hashes and coordinates, selects 1/N product instances by seed, exact class or instance index, and produces private full-resolution cyan/red ROI previews. See [ROI selection](ROI_SELECTION.md). User-side Windows checks still required.

**Implemented (Task 4):** bbox-derived `rect` / `soft_rect` float32 masks, inside-box feather, optional padding and maximum union for selected ROI. Local grayscale PNG, overlay PNG and JSON report. See [mask builder](MASKS.md).

**Implemented (Task 5):** deterministic local highlight compositing (`soft_glow`, `hotspot`) driven by Task-4 masks, with local effect/mask/overlay PNG outputs and a private JSON report. See [effects](EFFECTS.md).

**Task 6.1 (inspection only):** read-only local ComfyUI and optional exported workflow preflight; no custom nodes installed or graph modified. See [ComfyUI preflight](COMFYUI_PREFLIGHT.md).

**Task 6.2 (accepted):** `SAP Dataset Loader` reads a verified indexed JPEG+SEG pair from local ZIP files and outputs ComfyUI `IMAGE`, opaque `SAP_ANNOTATIONS`, and a short source summary. Includes a minimal workflow template. See [installation](COMFYUI_DATASET_LOADER.md).

**Task 6.3 (accepted):** `SAP ROI Selector` receives `SAP_ANNOTATIONS` and the source `IMAGE`, selects bbox instances via the pure-Python selector, and produces a cyan/red review preview plus opaque selection metadata. See [ComfyUI ROI Selector](COMFYUI_ROI_SELECTOR.md).

**Task 6.4 (accepted):** native `SAP Mask Builder` consumes an ROI selection and original IMAGE, returns a float32 `[1,H,W]` ComfyUI MASK and image overlay. See [ComfyUI Mask Builder](COMFYUI_MASK_BUILDER.md).

**Task 6.5 (accepted as a functional prototype):** `SAP Highlight Effect` consumes the original IMAGE and SAP MASK, applies the existing deterministic soft-glow/hotspot photometric engine, and returns an IMAGE plus a private short summary. See [ComfyUI Highlight Effect](COMFYUI_HIGHLIGHT_EFFECT.md).

**Task 6.6 (accepted as a technical prototype after user-side ComfyUI green-PASS smoke test):** `SAP QA Validator` checks source/modified RGB, effect mask and source bbox/selection invariants; emits BOOLEAN PASS/FAIL, JSON metrics and a green/red-border review image. Technical PASS never means approved-for-training. See [ComfyUI QA Validator](COMFYUI_QA_VALIDATOR.md).

**Task 6.7 (implemented, awaiting user-side visual acceptance):** `SAP Realistic Glare` generates a seeded, smooth elliptical or directional ROI-local light footprint with bbox-edge taper. Its **actual effect mask** is passed to the existing QA node; no source bbox/labels change. This is not a physically modeled glass reflection and no new `glint` labels are inferred. See [realistic glare](COMFYUI_REALISTIC_GLARE.md).

**Next planned tasks (not implemented):** annotation-preserving exporter (Stage 7), full QA/pilot acceptance (Stages 8–9). The current bbox-local soft glow is not approved for model training without visual review. See [roadmap](ROADMAP.md). Do not publish source data or reports of third-party datasets.

Detailed internal project plans and source dataset audits are **kept outside this public repository**.
