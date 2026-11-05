import io
import json
import pytest


def test_home_route(flask_client):
    """Verify home page loads with HTTP 200 and security headers."""
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert b"AI_Polymorphism_Detection_Sim" in resp.data
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "Content-Security-Policy" in resp.headers


def test_metrics_route(flask_client):
    """Verify metrics page displays KPIs and comparison table."""
    resp = flask_client.get("/metrics")
    assert resp.status_code == 200
    assert b"Accuracy (Ensemble)" in resp.data
    assert b"Hybrid Ensemble" in resp.data
    assert b"Group-Aware Split" in resp.data


def test_metrics_charts(flask_client):
    """Verify PNG chart endpoints return valid image/png."""
    fi_resp = flask_client.get("/metrics/feature_importance.png")
    assert fi_resp.status_code == 200
    assert fi_resp.mimetype == "image/png"

    cm_resp = flask_client.get("/metrics/confusion_matrix.png")
    assert cm_resp.status_code == 200
    assert cm_resp.mimetype == "image/png"


def test_playground_route(flask_client):
    """Verify playground page renders with sliders."""
    resp = flask_client.get("/playground")
    assert resp.status_code == 200
    assert b"Interactive Polymorphic Feature Playground" in resp.data


def test_dataset_route(flask_client):
    """Verify dataset browser displays schema and family counts."""
    resp = flask_client.get("/dataset")
    assert resp.status_code == 200
    assert b"Lineage Families" in resp.data
    assert b"Dataset Schema" in resp.data


def test_api_predict_valid(flask_client):
    """Verify valid JSON payload to /api/predict returns 200 with score and explanation."""
    payload = {
        "num_sections": 5,
        "avg_section_entropy": 6.8,
        "total_section_size": 250000,
        "imported_api_count": 20,
        "strings_count": 180,
        "junk_bytes_count": 3500,
        "resource_size": 45000,
    }
    resp = flask_client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "score" in data
    assert "pred" in data
    assert data["pred"] in ("benign", "malware_sim")
    assert "explanation" in data


def test_api_predict_validation_errors(flask_client):
    """Verify input validation handles non-JSON, out-of-bounds, and non-numeric inputs."""
    # 1. Non-JSON
    resp = flask_client.post("/api/predict", data="plain text")
    assert resp.status_code == 400

    # 2. Out of bounds entropy (> 8.0)
    resp = flask_client.post("/api/predict", json={"avg_section_entropy": 12.5})
    assert resp.status_code == 400
    assert "out of allowed bounds" in resp.get_json()["error"]

    # 3. Non-numeric value
    resp = flask_client.post("/api/predict", json={"num_sections": "invalid_number"})
    assert resp.status_code == 400
    assert "must be numeric" in resp.get_json()["error"]


def test_classify_upload(flask_client):
    """Verify artifact upload classification."""
    valid_record = {
        "id": "upload-test-1",
        "num_sections": 4,
        "section_sizes": [10000, 20000, 30000, 40000],
        "section_entropy": [3.5, 4.0, 3.8, 4.2],
        "imported_api_count": 65,
        "strings_count": 700,
        "junk_bytes_count": 200,
        "resource_size": 30000,
    }
    data = {
        "file": (io.BytesIO(json.dumps(valid_record).encode("utf-8")), "artifact.json")
    }
    resp = flask_client.post("/classify", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Classification Result" in resp.data

    # Test malformed JSON upload
    data_bad = {
        "file": (io.BytesIO(b"{bad json, incomplete"), "artifact.json")
    }
    resp_bad = flask_client.post("/classify", data=data_bad, content_type="multipart/form-data")
    assert resp_bad.status_code == 200
    assert b"Malformed JSON file" in resp_bad.data

