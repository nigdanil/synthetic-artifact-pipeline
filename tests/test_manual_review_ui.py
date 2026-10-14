"""Stage 8.3 manual UI tests use generated synthetic data only; no private dataset."""

from __future__ import annotations

import hashlib
import http.client
import json
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest
from test_batch_gallery import _create_run

from sapipeline.review.manual import ManualReviewError, read_state, save_review
from scripts.manual_review_ui import make_handler, make_page


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    _create_run(tmp_path, "run_alpha")
    return tmp_path


def payload(state: dict, verdict: str = "approved", **updates) -> dict:
    run = state["runs"][0]
    result = {
        "run_id": run["run_id"], "manifest_sha256": run["manifest_sha256"],
        "reviews_etag": state["reviews_etag"], "verdict": verdict,
        "reviewer": "Local QA", "reason": "", "checks": {
            "product_identity": True, "glare_naturalness": True, "bbox_consistency": True,
        },
    }
    result.update(updates)
    return result


def test_initial_state_pending_without_writing(project: Path) -> None:
    state = read_state(project)
    assert state["totals"] == {"runs": 1, "approved": 0}
    assert state["runs"][0]["review_status"] == "needs_review"
    assert state["runs"][0]["review"]["checks"]["bbox_consistency"] is False
    assert state["training_approved"] is False
    assert not (project / "reports/qa_acceptance/reviews.json").exists()


def test_approve_requires_checks_reviewer_and_qa(project: Path) -> None:
    state = read_state(project)
    for invalid in [
        payload(state, checks={"product_identity": True, "glare_naturalness": False, "bbox_consistency": True}),
        payload(state, reviewer=" "),
        payload(state, checks={"product_identity": 1, "glare_naturalness": True, "bbox_consistency": True}),
        payload(state, verdict="super_approved"),
    ]:
        with pytest.raises(ManualReviewError):
            save_review(project, invalid)
    assert not (project / "reports/qa_acceptance/reviews.json").exists()
    result = save_review(project, payload(state))
    assert result["verdict"] == "approved" and result["training_approved"] is False
    assert read_state(project)["runs"][0]["review_status"] == "approved"


def test_rejection_needs_human_reason(project: Path) -> None:
    state = read_state(project)
    with pytest.raises(ManualReviewError, match="at least 8"):
        save_review(project, payload(state, "rejected", reason="bad"))
    save_review(project, payload(state, "rejected", reason="Reflections look like hard stripes"))
    assert read_state(project)["runs"][0]["review_status"] == "rejected"


def test_atomic_update_preserves_other_reviews_and_backs_up(project: Path) -> None:
    _create_run(project, "run_beta", source="b" * 64)
    first = read_state(project)
    run_a = next(r for r in first["runs"] if r["run_id"] == "run_alpha")
    decision_a = payload(first, "approved")
    decision_a["run_id"] = run_a["run_id"]
    decision_a["manifest_sha256"] = run_a["manifest_sha256"]
    save_review(project, decision_a)
    reviews_file = project / "reports/qa_acceptance/reviews.json"
    before = reviews_file.read_bytes()
    new_state = read_state(project)
    run_b = next(r for r in new_state["runs"] if r["run_id"] == "run_beta")
    decision_b = payload(new_state, "rejected", reason="Distorted product lettering")
    decision_b["run_id"] = run_b["run_id"]
    decision_b["manifest_sha256"] = run_b["manifest_sha256"]
    save_review(project, decision_b)
    reviews = json.loads(reviews_file.read_text(encoding="utf-8"))["reviews"]
    assert {r["run_id"]: r["verdict"] for r in reviews} == {
        "run_alpha": "approved", "run_beta": "rejected",
    }
    backup = project / "reports/qa_acceptance/backups" / (
        "reviews_" + hashlib.sha256(before).hexdigest() + ".json"
    )
    assert backup.read_bytes() == before
    assert not list(reviews_file.parent.glob(".pending_reviews_*"))


def test_refuse_stale_tabs_and_stale_manifest(project: Path) -> None:
    state = read_state(project)
    save_review(project, payload(state, "needs_review", reviewer=""))
    with pytest.raises(ManualReviewError, match="Reviews changed"):
        save_review(project, payload(state))
    current = read_state(project)
    with pytest.raises(ManualReviewError, match="Manifest changed"):
        save_review(project, payload(current, manifest_sha256="0" * 64))


def test_stale_review_does_not_inherit_approved_checkbox(project: Path) -> None:
    first = read_state(project)
    save_review(project, payload(first))
    manifest = project / "outputs/sap_runs/run_alpha/manifest.json"
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    obj["some_new_field"] = "new version"
    manifest.write_text(json.dumps(obj), encoding="utf-8")
    # Changing the immutable manifest invalidates its existing comparison sheet.
    with pytest.raises(ManualReviewError, match="Stale or modified visual report"):
        read_state(project)
    # A newly rendered comparison sheet binds to the modified manifest, but
    # the previously approved human decision must NEVER carry over.
    metrics_file = project / "reports/visual_review/run_alpha/review_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
    current = read_state(project)
    row = current["runs"][0]
    assert row["review_status"] == "stale_review"
    assert row["review"]["verdict"] == "needs_review"
    assert not any(row["review"]["checks"].values())
    assert current["totals"]["approved"] == 0


def test_reject_tampered_bundle_and_symlinked_reviews(project: Path) -> None:
    before = read_state(project)
    img = project / "outputs/sap_runs/run_alpha/images/run_alpha.png"
    img.write_bytes(img.read_bytes() + b"illegal")
    with pytest.raises(ManualReviewError, match="SHA256"):
        save_review(project, payload(before))
    assert not (project / "reports/qa_acceptance/reviews.json").exists()
    img.write_bytes(img.read_bytes()[:-7])
    reviews = project / "reports/qa_acceptance/reviews.json"
    reviews.parent.mkdir(parents=True)
    outsider = project / "untrusted.json"
    outsider.write_text("{}")
    try:
        reviews.symlink_to(outsider)
    except OSError:
        pytest.skip("Symlinks unavailable on this filesystem")
    with pytest.raises(ManualReviewError, match="Symlink"):
        read_state(project)
    assert outsider.read_text() == "{}"


def test_ui_page_uses_strict_same_origin_script_and_no_training_approval() -> None:
    page = make_page("token", "nonce").decode("utf-8")
    assert 'nonce="nonce"' in page
    assert 'const csrf="token"' in page
    assert "Manual Review" in page
    assert "fetch('/api/review'" in page
    assert "training approval" in page
    assert "approved" in page and "rejected" in page and "needs_review" in page


def test_local_http_csrf_origin_and_save(project: Path) -> None:
    handler = make_handler(project, csrf="my-secret", nonce="page-nonce")
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_port

            def req(method: str, path: str, body: dict | None = None, headers=None):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                data = json.dumps(body).encode() if body is not None else None
                conn.request(method, path, body=data, headers=headers or {})
                res = conn.getresponse()
                payload_bytes = res.read()
                status = res.status
                conn.close()
                return status, payload_bytes

            status, page = req("GET", "/")
            assert status == 200 and b"/api/review" in page
            status, data = req("GET", "/api/state")
            assert status == 200
            state = json.loads(data)
            reqbody = payload(state)
            status, _ = req("POST", "/api/review", reqbody, {
                "Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}",
            })
            assert status == 403
            status, _ = req("POST", "/api/review", reqbody, {
                "Content-Type": "application/json", "X-CSRF-Token": "my-secret",
                "Origin": "https://evil.invalid",
            })
            assert status == 403
            status, body = req("POST", "/api/review", reqbody, {
                "Content-Type": "application/json", "X-CSRF-Token": "my-secret",
                "Origin": f"http://127.0.0.1:{port}",
            })
            assert status == 200 and json.loads(body)["verdict"] == "approved"
            status, data = req("GET", "/api/state")
            assert status == 200 and json.loads(data)["totals"]["approved"] == 1
            status, media = req("GET", "/media/run_alpha/original")
            assert status == 200 and media.startswith(b"\x89PNG")
            status, _ = req("GET", "/media/bad..name/original")
            assert status != 200
        finally:
            server.shutdown()
            thread.join(timeout=5)
