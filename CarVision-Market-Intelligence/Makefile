.PHONY: setup install lint test train eval predict serve dashboard

.PHONY: start-demo mlflow-demo

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

lint:
	. .venv/bin/activate && flake8 && black --check .

test:
	. .venv/bin/activate && pytest -q

train:
	. .venv/bin/activate && python main.py --mode train --config configs/config.yaml

eval:
	. .venv/bin/activate && python main.py --mode eval --config configs/config.yaml

predict:
	. .venv/bin/activate && python main.py --mode predict --config configs/config.yaml --input_json example_payload.json

serve:
	. .venv/bin/activate && uvicorn app.fastapi_app:app --reload --port 8000

dashboard:
	. .venv/bin/activate && streamlit run app/streamlit_app.py

start-demo:
	python -m pip install -r requirements.txt
	python main.py --mode train --config configs/config.yaml
	streamlit run app/streamlit_app.py --server.port 8501

mlflow-demo:
	python scripts/run_mlflow.py
