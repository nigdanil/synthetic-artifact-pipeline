"""These verify only the packaging scaffold, not image processing."""

from pathlib import Path

from sapipeline import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_public_docs_present() -> None:
    assert (ROOT / "docs" / "STATUS.md").is_file()
    assert (ROOT / "SECURITY.md").is_file()


def test_expected_project_folders_present() -> None:
    for rel in ("data/raw", "data/work", "workflows/existing", "outputs", "reports"):
        assert (ROOT / rel).is_dir(), rel
