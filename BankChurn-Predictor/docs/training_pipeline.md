# Pipeline de entrenamiento y evaluación — BankChurn Predictor

## CLI y modos soportados

`main.py` expone los siguientes modos principales:

- `train` — entrena modelo, evalúa en test holdout y guarda artefactos.
- `evaluate` — evalúa un modelo ya entrenado sobre un dataset dado.
- `predict` — genera predicciones para nuevos clientes y guarda CSV.
- `hyperopt` — ejecuta optimización de hiperparámetros.

Ejemplos (simplificados):

```bash
python main.py --mode train \
  --config configs/config.yaml \
  --seed 42 \
  --input Churn.csv \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl

python main.py --mode evaluate \
  --config configs/config.yaml \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --input Churn.csv

python main.py --mode predict \
  --config configs/config.yaml \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --input data/new_customers.csv \
  --output predictions.csv
```

## Detalle del pipeline de `train`

1. **Inicialización**
   - Se crea instancia de `BankChurnPredictor` con la configuración YAML.
   - Se fijan seeds (`numpy`, `random`, sklearn) para reproducibilidad.

2. **Carga de datos** (`load_data`)
   - Lee CSV de entrada.
   - Valida presencia de columna objetivo `Exited`.
   - Registra distribución de clases en logs.

3. **Preprocesamiento** (`preprocess_data`)
   - Elimina columnas listadas en `data.drop_columns`.
   - Separa `X` e `y`.
   - Construye `ColumnTransformer` con pipelines numérico/categórico.
   - Ajusta y transforma `X`, devolviendo `X_processed` como `DataFrame`.

4. **Split train/test**
   - `train_test_split` con parámetros de `training` en `configs/config.yaml`.
   - Estratificación opcional según `training.stratify`.

5. **Entrenamiento con CV** (`train`)
   - Crea modelo ensemble (LogReg + RandomForest) envuelto en `ResampleClassifier` si corresponde.
   - Ejecuta `StratifiedKFold(n_splits=5, shuffle=True)`.
   - Para cada fold:
     - entrena modelo en subset de train,
     - evalúa en val,
     - calcula F1, ROC-AUC, precision y recall.
   - Ajusta modelo final en todos los datos de entrenamiento.
   - Opcionalmente aplica calibración Platt (`CalibratedClassifierCV`).
   - Devuelve métricas agregadas (`*_mean`, `*_std`).

6. **Evaluación en test** (`evaluate`)
   - Calcula predicciones y probabilidades en el conjunto de test.
   - Métricas agregadas:
     - `f1_score`, `roc_auc`, `accuracy`, `precision`, `recall`.
   - Objetos adicionales:
     - `confusion_matrix` (2x2),
     - `classification_report` (dict anidado).

7. **Persistencia de artefactos**
   - Modelo y preprocesador: `save_model(model_path, preprocessor_path)`.
   - Resultados de entrenamiento: `results/training_results.json`:

```json
{
  "cv_results": {
    "f1_mean": 0.64,
    "f1_std": 0.02,
    "roc_auc_mean": 0.87,
    "roc_auc_std": 0.01
  },
  "test_results": {
    "metrics": {
      "f1_score": 0.637,
      "roc_auc": 0.867,
      "accuracy": 0.824,
      "precision": 0.54,
      "recall": 0.77
    },
    "confusion_matrix": [[1587, 206], [94, 313]],
    "classification_report": {"0": {"precision": ...}, "1": {"precision": ...}}
  }
}
```

   - Paquete combinado `models/model_v1.0.0.pkl` con `{"preprocessor": ..., "model": ..., "version": "1.0.0"}`.

## Integración con MLflow

- `scripts/run_mlflow.py`:
  - Lee `results/training_results.json`.
  - Registra métricas de CV (`cv_*`) y de test (`test_*`).
  - Loguea como artefactos `results/training_results.json`, `configs/config.yaml` y el paquete `models/model_v1.0.0.pkl` si existe.
  - Opcionalmente registra un `Pipeline` sklearn en el Model Registry (si backend lo permite).
  - Calcula métricas de negocio derivadas (p.ej. clientes en riesgo, clientes salvados y ROI proxy) a partir de la matriz de confusión y supuestos configurables (CLV medio y tasa de retención efectiva).

## Fairness y análisis por subgrupos

- Tests en `tests/test_fairness.py` usan datos sintéticos para evaluar gaps de recall entre subgrupos:
  - Por `Geography`: France vs Spain vs Germany.
  - Por `Gender`: Male vs Female.
- Lógica básica:
  - Entrenar un modelo simple (LogReg) sobre datos sintéticos con distribuciones controladas.
  - Calcular recall por subgrupo.
  - Verificar que el gap máximo de recall entre subgrupos esté por debajo de un umbral (0.3 en el dataset sintético).
- Estos tests no garantizan fairness en producción, pero sirven como guardrail mínimo en desarrollo.

## Evaluación y reporting

- Métricas de referencia (test):
  - F1-Score ≈ 0.637.
  - AUC-ROC ≈ 0.867.
  - Recall ≈ 0.77 (clase churn).
- Reportes adicionales:
  - Matriz de confusión e informe de clasificación (ver `results/training_results.json`).
  - Posible análisis SHAP/feature importance documentado en README/model_card, no cubierto directamente por este script.

## Uso típico en un flujo MLOps

1. Ejecutar entrenamiento controlado (`dvc repro` o comando `train` manual).
2. Revisar métricas técnicas y de negocio (incluyendo derivadas en MLflow).
3. Verificar tests (incluyendo `tests/test_fairness.py`).
4. Si los criterios de aceptación se cumplen, promover el modelo (`models/model_v1.0.0.pkl`) a entorno de serving (API FastAPI).
5. Monitorizar drift y performance; disparar reentrenos cuando la degradación supere umbrales acordados.
