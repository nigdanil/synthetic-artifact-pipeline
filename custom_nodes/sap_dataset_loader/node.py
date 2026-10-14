"""ComfyUI adapter for the read-only, hash-verified SAP dataset loader.

Installation requires copying only this extension directory into ComfyUI's
custom_nodes. On each invocation, project_root identifies the trusted source
checkout; ComfyUI's Python remains untouched and nothing is pip-installed.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path


class SAPDatasetLoader:
    """Read an indexed JPEG/SEG pair and expose IMAGE + opaque bbox annotations."""

    RETURN_TYPES = ("IMAGE", "SAP_ANNOTATIONS", "STRING", "SAP_SOURCE_CONTEXT")
    RETURN_NAMES = ("image", "annotations", "source_summary", "source_context")
    FUNCTION = "load"
    CATEGORY = "SAP/Dataset"
    DESCRIPTION = (
        "Read one indexed JPEG+SEG pair from local data/raw/*.zip, verify SHA-256, "
        "and return the encoded-grid RGB image and original bbox metadata. "
        "Never changes originals or writes output files."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_root": (
                    "STRING", {"default": "", "multiline": False,
                               "tooltip": "Absolute path to your trusted local SAP project directory"}
                ),
                "split": (["learning", "verification"],),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "image_member": (
                    "STRING", {"default": "", "multiline": False,
                               "tooltip": "Optional exact img/NAME.jpg inside ZIP; overrides sample_index"}
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        """Recheck bytes on each execution, rather than cache stale ZIP contents."""
        return math.nan

    @staticmethod
    def _core(project_root: str):
        if not isinstance(project_root, str) or not project_root.strip():
            raise ValueError("SAP project_root is required (absolute local path)")
        supplied = Path(project_root.strip()).expanduser()
        if not supplied.is_absolute():
            raise ValueError("SAP project_root must be an absolute local directory")
        root = supplied.resolve()
        src = (root / "src").resolve()
        if not (root / "pyproject.toml").is_file():
            raise ValueError("SAP project_root must be the existing project directory")
        if not (src / "sapipeline" / "dataset" / "loader.py").is_file():
            raise ValueError("SAP source package not found under project_root/src")
        # If a different SAP package is already in this process, do not mix versions.
        existing = sys.modules.get("sapipeline")
        if existing is not None:
            existing_file = Path(getattr(existing, "__file__", "") or "/invalid").resolve()
            if not existing_file.is_relative_to(src):
                raise RuntimeError("Another sapipeline version is already loaded in ComfyUI")
        # Insert only the trusted, explicitly selected source checkout at run time.
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return importlib.import_module("sapipeline.dataset"), root

    def load(self, project_root: str, split: str, sample_index: int, image_member: str):
        if split not in ("learning", "verification"):
            raise ValueError("split must be learning or verification")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if not isinstance(image_member, str):
            raise TypeError("image_member must be text")
        if image_member.strip() and sample_index != 0:
            raise ValueError("Set sample_index=0 when using an explicit image_member")

        dataset, root = self._core(project_root)
        # Optional dependencies belong to the ALREADY installed ComfyUI runtime.
        # Import only when the user queues this node, never at discovery time.
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "ComfyUI's Python needs its existing numpy/torch installation; "
                "do not install into ComfyUI automatically"
            ) from exc

        index = dataset.read_private_index(root / "data" / "work" / "index" / "dataset_index.json")
        entry = dataset.choose_index_entry(
            index, split=split, sample_index=sample_index,
            image_member=image_member.strip() or None,
        )
        pair = dataset.load_indexed_pair(index, entry, data_dir=root / "data" / "raw")
        rgb = np.asarray(pair.image, dtype=np.uint8)
        # ComfyUI IMAGE contract: CPU torch float32 [B,H,W,C], normalized to [0,1].
        image = torch.from_numpy(rgb.copy()).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
        summary = json.dumps(
            {
                "split": split,
                "size_px": [pair.image.width, pair.image.height],
                "bbox_count": len(pair.annotations.objects),
                "image_sha256": entry["image_sha256"],
                "orientation": "encoded-pixels-no-exif-transpose",
            },
            sort_keys=True,
        )
        # Return the AnnotationDocument in-process; do not serialize raw vendor
        # class names, paths, or source files into the public workflow JSON.
        # Opaque in-process provenance: no original filenames in public workflow JSON.
        return (image, pair.annotations, summary, {key: entry[key] for key in (
            "archive", "image_member", "annotation_member", "image_sha256",
            "annotation_sha256", "split",
        )})
