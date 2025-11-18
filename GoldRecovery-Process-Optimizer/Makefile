# GoldRecovery - Makefile

PY=python
PIP=pip
APP=app.fastapi_app:app
UVICORN=uvicorn

help:
	@echo "Targets disponibles: install, train, eval, predict, api, dashboard, test, format, lint"

install:
	$(PIP) install -r requirements.txt

_train_input=$$(python - <<'PY'
import yaml
try:
    cfg=yaml.safe_load(open('configs/config.yaml')) or {}
    print(cfg.get('data',{}).get('train_csv','gold_recovery_train.csv'))
except Exception:
    print('gold_recovery_train.csv')
PY
)

train:
	$(PY) main.py --mode train --config configs/config.yaml --input $(_train_input)

# Evaluacion sobre test
# Usa rutas del YAML si existen

_eval_input=$$(python - <<'PY'
import yaml; import sys
cfg=yaml.safe_load(open('configs/config.yaml'))
print(cfg.get('data',{}).get('test_csv','gold_recovery_test.csv'))
PY
)

eval:
	$(PY) main.py --mode evaluate --config configs/config.yaml --input $(_eval_input)

predict:
	$(PY) main.py --mode predict --model models/metallurgical_model.pkl --input gold_recovery_test.csv --output results/predictions.csv

api:
	$(UVICORN) $(APP) --host 0.0.0.0 --port 8000 --reload

serve: api

dashboard:
	streamlit run app/streamlit_dashboard.py --server.port 8501

start-demo:
	$(MAKE) install
	$(MAKE) train
	$(MAKE) api

mlflow-demo:
	$(PY) scripts/run_mlflow.py

check-drift:
	$(PY) monitoring/check_drift.py --ref gold_recovery_train.csv --cur gold_recovery_test.csv --out-json results/drift.json

test:
	pytest -q

format:
	black .

lint:
	flake8 .
