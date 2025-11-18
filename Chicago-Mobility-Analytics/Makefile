PY=python
UVICORN=uvicorn

.PHONY: install train eval api serve start-demo mlflow-demo check-drift test clean

install:
	pip install -r requirements.txt

train:
	$(PY) main.py --mode train --config configs/default.yaml --seed 42

eval:
	$(PY) main.py --mode eval --config configs/default.yaml

api:
	$(UVICORN) app.fastapi_app:app --host 0.0.0.0 --port 8000

serve: api

start-demo:
	$(MAKE) install
	$(MAKE) train
	$(PY) -m app.example_load

mlflow-demo:
	$(PY) scripts/run_mlflow.py

check-drift:
	$(PY) monitoring/check_drift.py --ref data/processed/trips_weather_features.csv --cur data/processed/trips_weather_features.csv --out-json artifacts/drift.json

clean:
	rm -rf artifacts models data/processed
