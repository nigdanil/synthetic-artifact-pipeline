"""Read-only diagnostics for the Python scaffold and optional ComfyUI directory.

This is NOT the dataset indexer, parser, or effect pipeline.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def label(present: bool) -> str:
    return "OK" if present else "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SAP setup check")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=None,
        help="Optional existing ComfyUI installation root (not modified)",
    )
    args = parser.parse_args()

    print(f"Project: {ROOT}")
    print(f"System: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print("\nPython modules:")
    for mod in ("sapipeline", "numpy", "PIL", "cv2", "pytest", "torch"):
        print(f"  {mod:<14} {label(find_spec(mod) is not None)}")

    print("\nDirectories:")
    for rel in ("data/raw", "data/work", "workflows/existing", "outputs", "reports"):
        path = ROOT / rel
        print(f"  {rel:<22} {label(path.is_dir())}")

    raw_dir = ROOT / "data" / "raw"
    local_inputs = sum(1 for path in raw_dir.iterdir() if path.is_file() and path.name != "README.md") if raw_dir.is_dir() else 0
    print(f"\nLocal data files detected (never published): {local_inputs}")

    if args.comfy_root:
        comfy = args.comfy_root.expanduser().resolve()
        print(f"\nExisting ComfyUI (read-only): {comfy}")
        for rel in ("main.py", "custom_nodes", ".venv/Scripts/python.exe"):
            item = comfy / rel
            print(f"  {rel:<29} {label(item.exists())}")
        print("  An alternative ComfyUI python interpreter layout may be valid.")

    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        print("\nERROR: Use Python 3.11–3.13 for this scaffold (3.12 recommended).")
        return 2
    if find_spec("sapipeline") is None:
        print("\nERROR: Package not importable. Try: python -m pip install -e .")
        return 2
    print("\nScaffold environment check finished (pipeline not implemented).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
