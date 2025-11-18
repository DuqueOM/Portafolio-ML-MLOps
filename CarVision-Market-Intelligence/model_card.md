# Model Card: CarVision Price Predictor

- **Model**: RandomForestRegressor inside sklearn Pipeline with preprocessing (impute + scale + one-hot)
- **Target**: price (USD)
- **Data**: `vehicles_us.csv` (used car listings). Source: educational dataset for analytics. See DATA_LICENSE.
- **Intended Use**: Educational/demo pricing intelligence. Not for real-world pricing decisions without further validation.
- **Metrics**: RMSE, MAE, MAPE, R2. Baseline: Dummy median.
- **Training**: Train/val/test split with fixed seed (42). See configs/config.yaml.
- **Limitations**:
  - Missing features (geography, trim, options) reduce accuracy.
  - Potential sampling bias; listings may not represent entire market.
  - No time-aware split; potential leakage from temporal effects.
- **Ethical Considerations**:
  - Avoid discriminatory pricing; sensitive attributes not used.
  - Dataset may reflect market biases; monitor for drift.
- **Risks & Misuse**:
  - Using predictions as exact valuations.
  - Applying outside data distribution.
- **Explainability**:
  - Notebook `notebooks/explainability_shap.ipynb` para análisis SHAP global/local sobre el modelo entrenado.
- **Versioning & Reproducibility**:
  - Config-driven. Artifacts stored under `artifacts/` with metrics and splits.
- **Contacts**: Maintainer Daniel Duque.

## Privacy & Data Governance

- El dataset de base (`vehicles_us.csv`) no contiene PII directa en este proyecto; IDs de anuncio/vendedor se eliminan para uso de portafolio.
- En un despliegue real, se recomienda anonimizar cualquier identificador adicional (cliente, vehículo, ubicación precisa) y evitar loggear payloads crudos en producción.
- Las predicciones de precio no deben combinarse con atributos sensibles externos (p. ej. barrio, proxies socioeconómicos) sin auditorías formales de fairness y cumplimiento normativo.
