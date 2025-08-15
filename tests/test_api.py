import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_schema(client):
    r = client.get("/schema")
    assert r.status_code == 200
    data = r.json()
    assert "numeric_features" in data
    assert "categorical_features" in data


def test_predict_minimal(client):
    schema = client.get("/schema").json()
    cols = schema["numeric_features"] + schema["categorical_features"] + schema.get("text_features", [])
    payload = {"features": {c: 0 if c in schema["numeric_features"] else "" for c in cols}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body and "probability" in body
