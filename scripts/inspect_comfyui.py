"""Read-only ComfyUI installation and workflow preflight; no ComfyUI imports.

Never runs external custom nodes or user workflows. Output is private and must
remain under reports/ (excluded from Git). Workflow input values are not read
into the resulting report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_WORKFLOW_BYTES = 8 * 1024 * 1024
MAX_NODES = 10_000
MAX_LINKS = 50_000


class PreflightError(ValueError):
    """Invalid local preflight input; no report should be written."""


def _bounded_json_file(path: Path) -> dict[str, Any]:
    """Read JSON without executing it; cap its size before allocating memory."""
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_WORKFLOW_BYTES + 1)
    except OSError as exc:
        raise PreflightError(f"Cannot read workflow file: {exc}") from exc
    if len(payload) > MAX_WORKFLOW_BYTES:
        raise PreflightError("Workflow JSON exceeds the 8 MiB inspection limit")
    try:
        parsed = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Invalid workflow JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PreflightError("Workflow JSON root must be an object")
    return parsed


def inspect_workflow(raw: dict[str, Any]) -> dict[str, Any]:
    """Describe node *types* and graph shape; never include widgets or inputs.

    Supports regular saved UI workflow with a `nodes` array and API-format
    workflow with node-ID keys and `class_type` values (possibly under `prompt`).
    """
    if not isinstance(raw, dict):
        raise PreflightError("Workflow JSON root must be an object")
    graph = raw
    if isinstance(raw.get("workflow"), dict) and "nodes" in raw["workflow"]:
        graph = raw["workflow"]
    if "nodes" in graph:
        nodes = graph["nodes"]
        links = graph.get("links", [])
        if not isinstance(nodes, list) or not isinstance(links, list):
            raise PreflightError("UI workflow nodes/links must be lists")
        if len(nodes) > MAX_NODES or len(links) > MAX_LINKS:
            raise PreflightError("Workflow exceeds inspection node/link limits")
        types: list[str] = []
        ids: set[int] = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict) or not isinstance(node.get("type"), str) or not node["type"]:
                raise PreflightError(f"UI node {i} has no valid type")
            identifier = node.get("id")
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                raise PreflightError(f"UI node {i} has an invalid numeric id")
            if identifier in ids:
                raise PreflightError(f"Duplicate UI node id {identifier}")
            ids.add(identifier)
            types.append(node["type"])
        invalid_links = 0
        for item in links:
            if isinstance(item, list) and len(item) >= 5 and (
                item[1] not in ids or item[3] not in ids
            ):
                invalid_links += 1
        counts = dict(sorted(Counter(types).items()))
        return {
            "format": "ui_workflow",
            "node_count": len(nodes),
            "link_count": len(links),
            "links_to_missing_nodes": invalid_links,
            "node_types": counts,
        }

    prompt = raw.get("prompt", raw)
    if not isinstance(prompt, dict):
        raise PreflightError("API workflow prompt must be an object")
    if len(prompt) > MAX_NODES:
        raise PreflightError("Workflow exceeds inspection node limit")
    types = []
    for identifier, node in prompt.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not node["class_type"]:
            raise PreflightError(f"No supported UI nodes or API class_type for node {identifier!r}")
        types.append(node["class_type"])
    return {
        "format": "api_prompt",
        "node_count": len(types),
        "link_count": None,
        "links_to_missing_nodes": None,
        "node_types": dict(sorted(Counter(types).items())),
    }


def inspect_installation(directory: Path) -> dict[str, Any]:
    """File presence and directory names only; do not import any ComfyUI code."""
    if not directory.is_dir():
        raise PreflightError("--comfyui-dir must point to an existing directory")
    markers = {name: (directory / name).is_file() for name in ("main.py", "requirements.txt", "pyproject.toml")}
    nodes = directory / "custom_nodes"
    if nodes.is_dir():
        try:
            packages = sorted(
                p.name for p in nodes.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
            )
        except OSError as exc:
            raise PreflightError(f"Cannot list custom_nodes directory: {exc}") from exc
    else:
        packages = []
    python_candidates = (
        ".venv/Scripts/python.exe",
        "venv/Scripts/python.exe",
        "python_embeded/python.exe",
        "python_embedded/python.exe",
    )
    runtime_candidates = [name for name in python_candidates if (directory / name).is_file()]
    warnings = []
    if not markers["main.py"]:
        warnings.append("No main.py in selected directory; check that this is the ComfyUI source root")
    if not nodes.is_dir():
        warnings.append("No custom_nodes directory found in selected directory")
    if not runtime_candidates:
        warnings.append("No conventional bundled Python executable found; runtime may be external")
    return {
        "markers": markers,
        "custom_nodes_directory_exists": nodes.is_dir(),
        "custom_node_directory_names": packages,
        "runtime_candidates_relative": runtime_candidates,
        "warnings": warnings,
    }


def _private_output(path: Path) -> Path:
    selected = path.expanduser().resolve()
    safe = (ROOT / "reports").resolve()
    if selected == safe or not selected.is_relative_to(safe):
        raise PreflightError("--out must stay inside this project's private reports/ directory")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a local ComfyUI installation without changing it")
    parser.add_argument("--comfyui-dir", required=True, type=Path, help="Existing ComfyUI source root")
    parser.add_argument("--workflow", type=Path, help="Optional exported UI/API workflow JSON (read-only)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/comfyui_preflight.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing private report")
    args = parser.parse_args(argv)

    try:
        output = _private_output(args.out)
        installation = inspect_installation(args.comfyui_dir)
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "inspection_mode": "read_only_no_code_execution",
            "comfyui": installation,
            "workflow": None,
            "workflow_supplied": args.workflow is not None,
            "next_action": "Export your existing ComfyUI workflow as JSON and rerun with --workflow" if args.workflow is None else "Review node types before writing the ComfyUI adapter",
        }
        if args.workflow is not None:
            report["workflow"] = inspect_workflow(_bounded_json_file(args.workflow))
        if output.exists() and not args.force:
            raise PreflightError("Private report already exists; use --force to overwrite it")
        output.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with tempfile.NamedTemporaryFile(prefix=".comfy-preflight-", suffix=".json", dir=output.parent, delete=False) as temporary:
            temporary.write(content)
            draft = Path(temporary.name)
        try:
            os.replace(draft, output)
        finally:
            draft.unlink(missing_ok=True)
        print(f"ComfyUI main.py: {'FOUND' if installation['markers']['main.py'] else 'NOT FOUND'}")
        print(f"Custom node directories: {len(installation['custom_node_directory_names'])}")
        print(f"Python runtime candidates: {len(installation['runtime_candidates_relative'])}")
        if report["workflow"] is not None:
            print(f"Workflow: {report['workflow']['format']} | nodes: {report['workflow']['node_count']}")
        else:
            print("Workflow: NOT SUPPLIED (can be added later)")
        print(f"Private report: {output}")
        return 0
    except (PreflightError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
