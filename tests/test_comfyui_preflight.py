"""Read-only ComfyUI preflight: no ComfyUI installation is required for tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspect_comfyui import (
    PreflightError,
    _bounded_json_file,
    _private_output,
    inspect_installation,
    inspect_workflow,
    main,
)


def test_ui_workflow_does_not_include_widget_secrets() -> None:
    workflow = {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["private.jpg", "TOKEN_SECRET"]},
            {"id": 2, "type": "PreviewImage", "widgets_values": ["PASSWORD"]},
        ],
        "links": [[9, 1, 0, 2, 0, "IMAGE"]],
    }
    result = inspect_workflow(workflow)
    assert result == {
        "format": "ui_workflow",
        "node_count": 2,
        "link_count": 1,
        "links_to_missing_nodes": 0,
        "node_types": {"LoadImage": 1, "PreviewImage": 1},
    }
    assert "TOKEN_SECRET" not in json.dumps(result)
    assert "private.jpg" not in json.dumps(result)


def test_ui_missing_link_target_flagged() -> None:
    result = inspect_workflow({"nodes": [{"id": 1, "type": "A"}], "links": [[0, 1, 0, 42, 0, "X"]]})
    assert result["links_to_missing_nodes"] == 1


def test_api_prompt_counts_classes_without_input_values() -> None:
    source = {"1": {"class_type": "LoadImage", "inputs": {"image": "confidential.jpg"}}, "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}
    result = inspect_workflow(source)
    assert result["format"] == "api_prompt"
    assert result["node_count"] == 2
    assert "confidential.jpg" not in json.dumps(result)


@pytest.mark.parametrize("payload", [{"nodes": "bad"}, {"nodes": [{"type": "A", "id": 1}, {"type": "B", "id": 1}]}, {"nodes": [{"id": 1}]}, {"prompt": []}])
def test_invalid_workflow_rejected(payload) -> None:
    with pytest.raises(PreflightError):
        inspect_workflow(payload)


def test_bounded_json_file(tmp_path: Path) -> None:
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    assert _bounded_json_file(source) == {"nodes": []}
    source.write_text("not json", encoding="utf-8")
    with pytest.raises(PreflightError):
        _bounded_json_file(source)


def test_installation_read_only(tmp_path: Path) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    nodes = root / "custom_nodes"
    nodes.mkdir()
    (nodes / "ExampleNode").mkdir()
    result = inspect_installation(root)
    assert result["markers"]["main.py"] is True
    assert result["custom_node_directory_names"] == ["ExampleNode"]
    assert not (root / "__pycache__").exists()


def test_invalid_installation_rejected(tmp_path: Path) -> None:
    with pytest.raises(PreflightError):
        inspect_installation(tmp_path / "absent")


def test_output_path_guard(tmp_path: Path, monkeypatch) -> None:
    from scripts import inspect_comfyui

    monkeypatch.setattr(inspect_comfyui, "ROOT", tmp_path)
    assert _private_output(tmp_path / "reports" / "preflight.json") == (tmp_path / "reports" / "preflight.json").resolve()
    with pytest.raises(PreflightError):
        _private_output(tmp_path / "secrets.json")


def test_cli_creates_private_report_without_comfyui_import(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts import inspect_comfyui

    monkeypatch.setattr(inspect_comfyui, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"nodes": [{"id": 1, "type": "TestNode", "widgets_values": ["my_private_file.jpg"]}]}), encoding="utf-8")
    target = tmp_path / "reports" / "comfy.json"
    assert main(["--comfyui-dir", str(comfy), "--workflow", str(workflow), "--out", str(target)]) == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["workflow"]["node_types"] == {"TestNode": 1}
    assert "my_private_file.jpg" not in json.dumps(report)
    assert main(["--comfyui-dir", str(comfy), "--out", str(target)]) == 2
    assert main(["--comfyui-dir", str(comfy), "--out", str(target), "--force"]) == 0
    assert "ComfyUI main.py" in capsys.readouterr().out
