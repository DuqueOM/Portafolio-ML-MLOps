from __future__ import annotations

import app.fastapi_app as api
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression


def test_health_and_predict_e2e():
    client = TestClient(api.app)

    # Health endpoint
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Prepare a tiny dummy model and inject
    X = pd.DataFrame([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], columns=["f0", "f1", "f2"])
    y = np.array([10.0, 1.0])
    m = LinearRegression().fit(X, y)
    api.models[1] = m

    payload = {"region": 1, "records": [{"f0": 1.0, "f1": 2.0, "f2": 3.0}]}
    r2 = client.post("/predict", json=payload)
    assert r2.status_code == 200
    out = r2.json()
    assert out["region"] == 1
    assert isinstance(out["predictions"], list) and len(out["predictions"]) == 1
