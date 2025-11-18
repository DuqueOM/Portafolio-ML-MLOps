# Data Card — Chicago Mobility Analytics

## Dataset Overview
- **Name:** Chicago taxi rides (educational subset)
- **Records:** ~6.4M rows in raw source; project uses curated Saturday sample (`moved_project_sql_result_07.csv`).
- **Features:** timestamps, pickup/dropoff zones, weather labels, derived temporal features.
- **Target:** `duration_seconds` (regression) + derived KPIs for streaming demo.

## Source & Licensing
- **Origin:** Open data portal / educational export bundled with TripleTen course.
- **License:** Educational/portfolio use (see `DATA_LICENSE`).
- **PII:** Trip IDs anonymized; no rider-level identifiers.

## Schema Snapshot
| Column | Type | Notes |
|--------|------|-------|
| `start_ts` | datetime | Trip start timestamp |
| `weather_conditions` | category | "Good" / "Bad" macro bucket |
| `duration_seconds` | float | Observed duration |
| `pickup_id` | int | Aggregated zone (if available) |
| `dropoff_id` | int | Aggregated zone |
| `distance_km` | float | Great-circle approximation |
| `hour`, `day_of_week`, `is_weekend` | derived | Feature engineering outputs |

## Splits & Versioning
- Default: 70/15/15 split defined in `configs/default.yaml` (seed=42).
- Processed features saved under `data/processed/` via `make train`.
- Geo assets versioned through `dvc.yaml` (CSV → GeoParquet stage).

## Quality Considerations
- Raw dataset contains outliers (duration <= 0, extreme weather labels) filtered in preprocessing.
- Weather labels coarse → limited granularity for causal claims.
- Sample limited to Saturdays; not representative of weekdays.

## Bias & Ethical Notes
- Spatial coverage limited to zones with data → may underrepresent certain neighborhoods.
- Travel times affected by socio-economic & infrastructure differences not modeled explicitly.
- Use predictions as estimates; include uncertainty when communicating externally.

## Privacy & Data Governance

- Los datos originales del portal abierto se publican con IDs anonimizados; este proyecto trabaja sobre agregados a nivel de viaje/zona.
- En contextos operativos, se debe limitar el acceso a datos de trayectos individuales y aplicar técnicas de agregación/aleatorización para evitar reidentificación de patrones sensibles.
- Cualquier uso secundario (p. ej. análisis de zonas de baja renta) debe pasar por una revisión de impacto ético y cumplimiento regulatorio.

## Refresh Strategy
- Pull latest trips from Chicago open data API monthly; regenerate features.
- Recompute drift metrics (`monitoring/check_drift.py`) comparing recent vs historical distributions.
- Keep GeoParquet stage updated via `dvc repro` when raw geodata changes.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `Chicago-Mobility-Analytics/`
