# Model Card: Gaming Market Success Classifier

- **Model**: RandomForestClassifier within a preprocessing pipeline (impute + scale + one-hot)
- **Task**: Binary classification of commercial success of a video game (>= 1.0M global sales)
- **Dataset**: Historic video games dataset (1980-2016). Columns include platform, year_of_release, genre, critic_score, user_score, rating, regional sales.
- **Owners**: Daniel Duque

## Intended Use
- Support product/marketing decisions with a probability of success for planned titles.
- Operates on metadata available before launch (no sales fields as inputs).

## Factors
- Inputs: platform, genre, year_of_release, critic_score, user_score, rating.
- Sensitive attributes: none directly, but proxies could exist (e.g., region genres).

## Metrics
- Primary: F1. Secondary: Precision, Recall, Accuracy, ROC-AUC.
- Report metrics via `make eval` (written to `artifacts/metrics/`).

## Evaluation Data
- 80/20 train/test split with stratification on the target.
- Success threshold at 1.0 (millions of units).

## Ethical Considerations
- Risk of popularity bias: past market preferences can penalize innovative/new genres or platforms.
- Temporal drift: older eras dominate; ensure retraining on fresh data.
- Do not use model outputs deterministically for greenlighting decisions.
- Platform bias: el modelo puede favorecer plataformas históricamente exitosas, reduciendo la probabilidad de greenlighting en plataformas emergentes.
- Regional bias: patrones de ventas por región pueden llevar a sobredimensionar mercados tradicionales (NA/EU) frente a otros con menos datos.
- Genre bias: juegos de géneros con baja representación histórica pueden ser sistemáticamente infravalorados en comparación con Action/Sports/Shooter.

## Risks & Limitations
- Dataset missingness for scores/ratings; imputations may bias associations.
- Dataset ends in 2016; predictions for current consoles require retraining.
- OOD risk: unseen platforms/genres → handled with OHE(handle_unknown=ignore) but calibration may suffer.

## Caveats and Recommendations
- Regularly retrain with latest releases; monitor performance by era/platform.
- Track data drift and class balance; adjust threshold and class_weight accordingly.
- Complement with causal/business analyses rather than relying solely on correlations.

## Privacy & Data Governance

- El dataset (`games.csv`) contiene información agregada a nivel de título (ventas, ratings), sin PII de jugadores.
- En un despliegue real, se debe garantizar que cualquier fuente adicional (telemetría de usuario, compras in-game) se anonimice y se trate bajo marcos de privacidad aplicables (p. ej. GDPR/CCPA).
- Evitar registrar payloads completos de scoring en logs y limitar el acceso a datos de entrenamiento a equipos con necesidad justificada.
