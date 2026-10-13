# Status of the public scaffold (2026-10-13)

The repository contains Python packaging, placeholder modules, empty runtime directories, a read-only environment check and initial smoke tests. It does not contain implemented augmentation or functional ComfyUI custom nodes.

**Implemented (Task 1):** read-only ZIP indexer, JPEG/SEG pairing, raw SHA-256 hashes, exact-duplicate/cross-split grouping and private JSON reports. See [dataset indexing](DATASET_INDEXING.md). Runtime validation on the user's Windows machine is still required.

**Next planned task (not implemented):** canonical `.seg` / YOLO TXT annotation parser with bbox geometry validation. Do not publish source data or reports of third-party datasets.

Detailed internal project plans and source dataset audits are **kept outside this public repository**.
