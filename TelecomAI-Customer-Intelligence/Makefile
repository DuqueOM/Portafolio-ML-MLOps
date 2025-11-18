PYTHON=python
CONFIG=configs/config.yaml
ARTIFACTS=artifacts

.PHONY: train eval predict serve test clean start-demo mlflow-demo

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) main.py --mode train --config $(CONFIG)

eval:
	$(PYTHON) main.py --mode eval --config $(CONFIG)

predict:
	$(PYTHON) main.py --mode predict --config $(CONFIG) --input_csv users_behavior.csv --output_path $(ARTIFACTS)/predictions.csv

serve:
	uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

start-demo:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) main.py --mode train --config $(CONFIG)
	docker compose up --build -d

mlflow-demo:
	$(PYTHON) scripts/run_mlflow.py

test:
	pytest -q

clean:
	rm -rf $(ARTIFACTS)
