# Status of the public scaffold (2026-10-13)

The repository contains Python packaging, placeholder modules, empty runtime directories, a read-only environment check and initial smoke tests. It does not contain implemented augmentation or functional ComfyUI custom nodes.

**Implemented (Task 1):** read-only ZIP indexer, JPEG/SEG pairing, raw SHA-256 hashes, exact-duplicate/cross-split grouping and private JSON reports. See [dataset indexing](DATASET_INDEXING.md). Confirmed locally on Windows: both archives indexed; Python environment/tests verified.

**Implemented (Task 2):** canonical `.seg` / YOLO TXT bbox parsers, strict geometry checks, explicit class mapping, lossless field preservation on SEG rewrite, and local-only bbox previews. See [annotations](ANNOTATIONS.md). Verify on the user's Windows environment before accepting Task 2.

**Implemented (Task 3):** local ZIP image/SEG loader verifies content hashes and coordinates, selects 1/N product instances by seed, exact class or instance index, and produces private full-resolution cyan/red ROI previews. See [ROI selection](ROI_SELECTION.md). User-side Windows checks still required.

**Implemented (Task 4):** bbox-derived `rect` / `soft_rect` float32 masks, inside-box feather, optional padding and maximum union for selected ROI. Local grayscale PNG, overlay PNG and JSON report. See [mask builder](MASKS.md).

**Implemented (Task 5):** deterministic local highlight compositing (`soft_glow`, `hotspot`) driven by Task-4 masks, with local effect/mask/overlay PNG outputs and a private JSON report. See [effects](EFFECTS.md).

**Next planned task (not implemented):** connect Python logic to the user's existing ComfyUI graph / exporter layer. Do not publish source data or reports of third-party datasets.

Detailed internal project plans and source dataset audits are **kept outside this public repository**.
