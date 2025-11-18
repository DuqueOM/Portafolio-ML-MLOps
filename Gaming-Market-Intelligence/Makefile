PY=python
CONFIG=configs/config.yaml

.PHONY: install-deps train eval predict api test format lint

install-deps:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) main.py --mode train --config $(CONFIG)

eval:
	$(PY) main.py --mode eval --config $(CONFIG)

predict:
	$(PY) main.py --mode predict --config $(CONFIG) --payload '{"platform":"PS4","genre":"Action","year_of_release":2015,"critic_score":85,"user_score":8.2,"rating":"M"}'

api:
	uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

serve: api

test:
	pytest -q

format:
	black .

lint:
	flake8 .

start-demo:
	$(MAKE) install-deps
	$(MAKE) train
	$(MAKE) api

mlflow-demo:
	$(PY) scripts/run_mlflow.py

check-drift:
	$(PY) monitoring/check_drift.py --ref artifacts/processed/sample_ref.csv --cur artifacts/processed/sample_ref.csv --out-json artifacts/metrics/drift.json
