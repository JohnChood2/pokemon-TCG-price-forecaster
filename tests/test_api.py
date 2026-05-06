"""Smoke test for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from pokemon_forecaster.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body


def test_predict_returns_400_with_no_history():
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={"card_id": "does-not-exist", "variant": "holofoil", "horizon_days": 7},
        )
        assert r.status_code == 400
