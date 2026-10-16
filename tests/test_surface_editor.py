"""Regression tests for Surface Editor's local image inspection endpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def editor_server():
    from http.server import ThreadingHTTPServer

    path = Path(__file__).resolve().parents[1] / "scripts/surface_editor.py"
    spec = importlib.util.spec_from_file_location("sap_surface_editor_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.SurfaceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def inspect(port: int, content: bytes):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(
            "POST", "/inspect", body=content,
            headers={"Content-Type": "application/octet-stream",
                     "Origin": f"http://127.0.0.1:{port}"},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_valid_source_image_returns_metadata(editor_server, fmt):
    buffer = BytesIO()
    Image.new("RGB", (80, 60), (32, 64, 96)).save(buffer, format=fmt)
    source = buffer.getvalue()
    status, result = inspect(editor_server, source)
    assert status == 200
    assert result == {
        "sha256": hashlib.sha256(source).hexdigest(),
        "width": 80,
        "height": 60,
    }


def test_corrupt_source_returns_json_error(editor_server):
    status, result = inspect(editor_server, b"invalid image bytes")
    assert status == 400
    assert "error" in result


def test_rotated_source_is_rejected(editor_server):
    buffer = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 60)).save(buffer, format="JPEG", exif=exif)
    status, result = inspect(editor_server, buffer.getvalue())
    assert status == 400
    assert "EXIF" in result["error"]
