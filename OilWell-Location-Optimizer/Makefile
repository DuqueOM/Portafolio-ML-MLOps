PY=python
UVICORN=uvicorn
CFG=configs/default.yaml

.PHONY: install train eval predict api test lint clean

install:
	pip install -r requirements.txt

train:
	$(PY) main.py --mode train --config $(CFG)

# compute RMSE and bootstrap risk per región
eval:
	$(PY) main.py --mode eval --config $(CFG)

api:
	$(UVICORN) app.fastapi_app:app --host 0.0.0.0 --port 8000

# alias
serve: api

predict-example:
	$(PY) main.py --mode predict --config $(CFG) --region 1 --payload '{"records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}'

# quick demo: install deps, train, then serve API
start-demo:
	$(MAKE) install
	$(MAKE) train
	$(MAKE) api

# MLflow demo logging (optional if mlflow installed)
mlflow-demo:
	$(PY) scripts/run_mlflow.py

# Monitoring drift example (uses KS/PSI and optional Evidently)
check-drift:
	$(PY) monitoring/check_drift.py --ref geo_data_1.csv --cur geo_data_2.csv --cols f0 f1 f2 --out-json artifacts/drift.json --report-html artifacts/drift_report.html

test:
	pytest -q

lint:
	flake8 . || true

clean:
	rm -rf artifacts
