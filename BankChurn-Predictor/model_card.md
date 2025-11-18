# Model Card — BankChurn Predictor

- Model: VotingClassifier (LogReg + RandomForest) with optional resampling and Platt calibration
- Target: `Exited` (1=churn, 0=retained)
- Features: categorical: `Geography`, `Gender`; numerical: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Version: 1.0.0

## Intended Use
- Anticipar abandono de clientes y priorizar intervenciones de retención.
- No usar como única fuente de decisión; requiere validación humana y políticas de contacto.

## Data
- Fuente: dataset sintético Beta Bank (10k filas). Columna objetivo `Exited`.
- Preprocesamiento: drop de IDs, OneHot para categorías, escalado numérico.
- Divisiones: estratificadas train/val/test.

## Performance (demo)
- CV (5-fold): f1_mean, roc_auc_mean registrados en `results/training_results.json`.
- Test: `f1_score`, `roc_auc`, `precision`, `recall`, `accuracy`.

## Calibration
- Platt (sigmoid) sobre el modelo final (cv="prefit").

## Limitations and Risks
- Desbalance 80/20 limita recall/precision simultáneamente.
- Posible drift temporal (edad/actividad) requiere monitoreo periódico.
- Riesgo de sesgo por `Geography` y `Age`.

## Fairness Note
- Se incluyen tests de fairness por geografía y género (gap de recall tolerancia < 0.3 en datos sintéticos). Monitorear en datos reales.

## Ética y Fairness

- **Supuestos de uso**
  - Dataset sintético (`Churn.csv`) usado para entrenamiento; no contiene PII real.
  - El modelo se diseña como herramienta de soporte para priorizar contactos de retención, no como filtro automático de servicio.
  - Las decisiones finales deben incorporar políticas de riesgo, cumplimiento y supervisión humana.
- **Riesgos identificados**
  - Posible sesgo por geografía (`Geography`) y edad (`Age`), al influir fuertemente en la probabilidad de churn.
  - Riesgo de discriminar sistemáticamente a clientes mayores o de ciertas regiones si se usan las predicciones sin controles.
  - Uso del modelo fuera del dominio de entrenamiento (otros bancos/países) puede amplificar distorsiones.
- **Mitigaciones implementadas (nivel demo)**
  - Tests sintéticos de fairness en `tests/test_fairness.py` para controlar gaps de recall entre subgrupos de `Geography` y `Gender`.
  - Métrica principal F1/Recall para reducir el riesgo de ignorar sistemáticamente clientes en riesgo (clase minoritaria).
  - Documentación explícita de riesgos y limitaciones en esta model card y en `data_card.md`.
- **Mitigaciones recomendadas en producción**
  - Auditorías periódicas de performance por segmento (edad, geografía, género, productos).
  - Ajuste de umbrales y/o reentrenamiento si se detectan gaps de performance injustificados.
  - Inclusión de stakeholders de negocio, legal y riesgo en la definición de políticas de uso y exclusiones.

## SLO
- Disponibilidad de API: 99.5% (objetivo).
- Latencia P95 inferencia: < 50 ms (CPU).
- Reentrenamiento: mensual o ante drift significativo.

## Maintenance
- Artefactos en `models/` (incluye `model_v1.0.0.pkl` empaquetado) y logs en `results/`.
- MLflow opcional para registro de runs y (si disponible) Model Registry.
