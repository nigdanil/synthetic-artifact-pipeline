# Status of the public scaffold (2026-10-13)

The repository contains Python packaging, placeholder modules, empty runtime directories, a read-only environment check and initial smoke tests. It does not contain implemented augmentation or functional ComfyUI custom nodes.

**Implemented (Task 1):** read-only ZIP indexer, JPEG/SEG pairing, raw SHA-256 hashes, exact-duplicate/cross-split grouping and private JSON reports. See [dataset indexing](DATASET_INDEXING.md). Confirmed locally on Windows: both archives indexed; Python environment/tests verified.

**Implemented (Task 2):** canonical `.seg` / YOLO TXT bbox parsers, strict geometry checks, explicit class mapping, lossless field preservation on SEG rewrite, and local-only bbox previews. See [annotations](ANNOTATIONS.md). Verify on the user's Windows environment before accepting Task 2.

**Next planned task (not implemented):** source-image loading and deterministic ROI selection. Do not publish source data or reports of third-party datasets.

Detailed internal project plans and source dataset audits are **kept outside this public repository**.
