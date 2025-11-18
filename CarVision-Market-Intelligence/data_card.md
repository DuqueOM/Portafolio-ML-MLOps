# Data Card — CarVision Market Intelligence

## Dataset Overview
- **Name:** vehicles_us (used car listings)
- **Records:** ~51,525 listings
- **Features:** mix of categorical and numerical attributes (model, year, condition, odometer, etc.)
- **Target:** `price` (USD)

## Source & Licensing
- **Origin:** Educational dataset distributed with the project (TripleTen). Resides at `vehicles_us.csv`.
- **License:** Educational/portfolio use (see `DATA_LICENSE`).
- **PII:** None. Listing IDs and seller info were removed.

## Schema Summary (columns más usados)
| Column | Type | Notes |
|--------|------|-------|
| `price` | float | Target variable in USD |
| `model_year` | int | Year of manufacture |
| `model` | category | Model name |
| `condition` | category | Condition string (e.g., good, like new) |
| `cylinders` | category | Engine cylinders |
| `fuel` | category | Fuel type |
| `odometer` | float | Mileage |
| `transmission` | category | Transmission type |
| `drive` | category | Drivetrain |
| `size` | category | Vehicle size |
| `type` | category | Body type |
| `paint_color` | category | Exterior color |
| `is_4wd` | int | 1 if 4WD flagged |
| `days_listed` | int | Days on market |

## Splits & Versioning
- Default split: 70% train / 15% validation / 15% test (seed configurable in `configs/config.yaml`).
- Processing pipeline saves cleaned datasets under `data/processed/`.
- Artifacts and metrics stored in `artifacts/` after `make train`/`make eval`.

## Quality Considerations
- Missing values in several columns (model year, cylinders, paint color) — imputed during preprocessing.
- Outliers in price and odometer filtered via `data/preprocess.py` thresholds.
- Data collected across different years; no strict temporal split (potential leakage for time-dependent tasks).

## Bias & Ethical Notes
- Dataset represents US online listings → may not generalize to other regions.
- Condition descriptions are human-entered and subjective.
- Potential sampling bias toward certain brands/geographies.

## Coverage & Usage Assumptions
- **Geographic coverage:** anuncios de vehículos usados en Estados Unidos; no se garantiza representatividad para otros países o mercados locales.
- **Temporal coverage:** múltiples años de publicación mezclados; no se modela explícitamente el efecto tiempo en el dataset base.
- **Educational use:** el dataset se incluye con fines de aprendizaje/portafolio; no debe usarse como fuente única para pricing real sin calibración adicional y datos propios del negocio.
- **Limitations:** faltan variables relevantes (ubicación exacta, impuestos locales, incentivos, historial de mantenimiento), por lo que las predicciones deben interpretarse como aproximaciones, no tasaciones oficiales.

## Privacy & Data Governance

- No se incluyen identificadores personales de compradores/vendedores en este proyecto; el CSV se usa con fines educativos.
- En contextos reales, se deben anonimizar IDs y desactivar el logging de datos crudos en endpoints de inferencia.
- Cualquier unión con datasets externos que contengan atributos sensibles (p.ej., nivel socioeconómico de zonas) debe pasar por una revisión de cumplimiento y fairness.

## Refresh Strategy
- Re-download or scrape latest listings periodically (monthly/quarterly).
- Recompute descriptive stats and drift checks (`monitoring/check_drift.py`).
- Log dataset metadata (date, source) in MLflow/DVC when available.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `CarVision-Market-Intelligence/`
