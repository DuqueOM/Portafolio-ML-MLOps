# Data Card — GoldRecovery Process Optimizer

## Dataset Overview
- **Name:** gold_recovery_train.csv / gold_recovery_test.csv / gold_recovery_full.csv
- **Records:** ~19,000 lotes (train) + 4,900 (test)
- **Features:** Variables de proceso de flotación/limpieza (concentraciones Au/Ag/Pb, caudales, rougher/cleaner outputs, etc.)
- **Target:** `final.output.recovery` (%)

## Source & Licensing
- **Origin:** TripleTen educational dataset (Kaggle metallurgy challenge adaptado).
- **License:** Uso educativo/portafolio (ver `DATA_LICENSE`).
- **PII:** Ninguna. Datos industriales anonimizados.

## Schema Snapshot (subset)
| Column | Type | Notes |
|--------|------|-------|
| `rougher.output.concentrate_au` | float | g/t Au en rougher concentrate |
| `primary_cleaner.output.concentrate_au` | float | g/t Au en cleaner concentrate |
| `rougher.output.concentrate_ag` | float | g/t Ag |
| `primary_cleaner.output.concentrate_pb` | float | g/t Pb |
| `rougher.input.feed_au` | float | ley de alimentación |
| `rougher.input.feed_size` | float | granulometría |
| `final.output.recovery` | float | Target (%) |

> Nota: más de 80 columnas adicionales. Los scripts seleccionan subset relevante vía `configs/features.yaml` cuando aplica.

## Splits & Versioning
- Entrenamiento: dataset train original con cross-validation. Hold-out provisto (`gold_recovery_test.csv`).
- Artefactos almacenados en `models/` y `artifacts/` (`metrics.json`, `safeties.json`, etc.).
- Features/postprocesamiento generados en runtime; usar `make train` para reproducir.

## Data Quality Considerations
- Valores faltantes abundantes; se aplican imputaciones y filtros de QA (`data/preprocess.py`).
- Algunas columnas son altamente correlacionadas → se aplican selecciones para evitar multicolinealidad.
- Targets cercanos a 0 o 100 pueden generar outliers en sMAPE.

## Bias & Ethical Notes
- Dataset corresponde a una sola planta/proceso → no generaliza automáticamente a otros circuitos.
- Cambios mineralógicos (drift) son críticos; se requiere monitoreo (ver `monitoring/check_drift.py`).
- Optimización sin validación humana puede poner en riesgo seguridad/ambiente → usar `safety_checks.py`.

## Privacy & Data Governance

- Aunque los archivos de entrada del reto están anonimizados, en entornos de planta los datos de proceso suelen ser confidenciales.
- Se recomienda aislar datasets de entrenamiento en entornos controlados, con acceso restringido y sin exportaciones ad-hoc a terceros.
- En integraciones API, evitar exponer detalles finos de proceso y limitar las respuestas a métricas relevantes para el usuario final.

## Refresh Strategy
- Actualizar con datos recientes del laboratorio/planta (idealmente semanal/mensual).
- Registrar versión (fecha + campaña) en `docs/operations.md` y MLflow.
- Recalibrar límites de seguridad en `safety_checks.RANGES` al refrescar datos.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `GoldRecovery-Process-Optimizer/`
