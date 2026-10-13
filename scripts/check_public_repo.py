"""Fail-closed preflight for files tracked or staged for a public SAP repo.

This cannot detect every kind of confidential business information. Inspect the diff too.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = (
    "data/raw/", "data/work/", "data/pilot/", "outputs/", "reports/",
    "workflows/existing/", "configs/class_mappings/", "docs/reference/",
    "docs/internal/", "docs/private/", "private/", "raw/", "datasets/",
)
ALLOWED_PLACEHOLDERS = {
    "data/raw/README.md", "data/work/.gitkeep", "data/pilot/.gitkeep",
    "outputs/.gitkeep", "reports/.gitkeep", "workflows/existing/README.md",
    "configs/class_mappings/README.md",
}
BLOCKED_EXTS = {
    ".seg", ".zip", ".7z", ".rar", ".tar", ".gz", ".jpg", ".jpeg",
    ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".mp4", ".mov",
    ".wav", ".mp3", ".csv", ".tsv", ".parquet", ".sqlite", ".sqlite3",
    ".db", ".pt", ".pth", ".ckpt", ".safetensors", ".p12", ".pfx",
    ".pem", ".key", ".jsonl", ".ndjson",
}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s]+"),
]


def main() -> int:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False,
    )
    if proc.returncode:
        print("FAIL: git repository not initialized or not accessible")
        return 2
    paths = [p.decode("utf-8", errors="replace") for p in proc.stdout.split(b"\0") if p]
    if not paths:
        print("FAIL: no tracked/staged files; run git add on intended public files first")
        return 2
    errors: list[str] = []
    for name in paths:
        relative = name.replace("\\", "/")
        path = ROOT / name
        if relative in ALLOWED_PLACEHOLDERS:
            continue
        if any(relative.startswith(prefix) for prefix in BLOCKED_DIRS):
            errors.append(f"Blocked local-only path: {relative}")
            continue
        if path.suffix.lower() in BLOCKED_EXTS or path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"Blocked file type: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"Symbolic link: {relative}")
            continue
        # Inspect the content *staged in the Git index*, not only the working tree.
        # A file can be changed after git add; git commit would use the older staged bytes.
        staged = subprocess.run(
            ["git", "show", f":{name}"], cwd=ROOT, capture_output=True, check=False,
        )
        if staged.returncode:
            errors.append(f"Cannot inspect Git index version: {relative}")
            continue
        if len(staged.stdout) > 1024 * 1024:
            errors.append(f"Unreviewed staged file > 1 MiB: {relative}")
            continue
        content = staged.stdout.decode("utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"Possible credential/private home path in: {relative}")
                break
    # Old commits may still be published by a normal `git push`, even when
    # current .gitignore and the staged index are clean.
    history = subprocess.run(
        ["git", "log", "--all", "--pretty=format:", "--name-only"],
        cwd=ROOT, capture_output=True, check=False, text=True,
    )
    if history.returncode == 0:
        past_paths = [line.strip() for line in history.stdout.splitlines() if line.strip()]
        for past in past_paths:
            if any(past.startswith(prefix) for prefix in BLOCKED_DIRS) and past not in ALLOWED_PLACEHOLDERS:
                errors.append(f"Private path appears in Git history: {past}")
                break
            if Path(past).suffix.lower() in BLOCKED_EXTS:
                errors.append(f"Blocked data file appears in Git history: {past}")
                break
    if errors:
        print("FAIL: public publication audit found potential leaks:")
        for error in errors:
            print(" -", error)
        return 1
    print(f"OK: {len(paths)} tracked/staged paths checked. Manual review still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
