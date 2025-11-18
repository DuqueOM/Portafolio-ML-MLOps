# Data Card — Gaming Market Intelligence

## Dataset Overview
- **Name:** Historic video game sales dataset (1980–2016)
- **Records:** 16,715 games
- **Features:** platform, year_of_release, genre, critic_score, user_score, rating, regional/global sales
- **Target:** `is_successful` (binary label >= 1M global sales) for classifier; `global_sales` for regression-style notebooks

## Source & Licensing
- **Origin:** Kaggle/TripleTen educational dataset packaged with the project (`games.csv`).
- **License:** Educational/portfolio use (DATA_LICENSE).
- **PII:** None. All data aggregated at product level.

## Schema Snapshot
| Column | Type | Notes |
|--------|------|-------|
| `name` | string | Game title |
| `platform` | category | Console/PC platform |
| `year_of_release` | int | 1980–2016 |
| `genre` | category | Action, Sports, etc. |
| `publisher` | string | Publisher name |
| `na_sales`, `eu_sales`, `jp_sales`, `other_sales`, `global_sales` | float | Millions of units |
| `critic_score`, `critic_count` | float/int | Review aggregates |
| `user_score`, `user_count` | float/int | User reviews |
| `rating` | category | ESRB rating |

## Splits & Versioning
- Default 80/20 stratified split on `is_successful` (seed=42). Configurable via `configs/config.yaml`.
- Processed datasets stored under `data/processed/` when running pipeline scripts.
- Artifacts (metrics, models) saved in `artifacts/` per run.

## Data Quality Considerations
- Missing values in scores/ratings; imputed (median/mode) or dropped per pipeline.
- Some rows contain `tbd` user scores (converted to NaN).
- Sales data aggregated — potential rounding/truncation errors.

## Bias & Ethical Notes
- Historical bias toward dominant platforms/genres; may underrepresent indie/mobile markets.
- Dataset stops at 2016 — not reflective of modern consoles (PS5, Switch) or live-service trends.
- Cultural biases across regions: treat regional insights carefully.
- Platform bias: ventas históricas favorecen plataformas líderes (PS4/Xbox/PC); modelos entrenados pueden infravalorar plataformas emergentes o nicho.
- Regional bias: NA/EU concentran gran parte de las ventas; insights por región pueden reforzar estrategias que ignoren mercados menos representados.
- Genre bias: géneros históricamente exitosos (Action/Sports/Shooter) pueden ser sobrepriorizados, penalizando títulos de géneros más nicho o innovadores.

## Privacy & Data Governance

- Los datos están agregados a nivel de juego; no contienen identificadores de jugadores.
- Si se combinan con fuentes operativas (telemetría, CRM), se deben aplicar técnicas de anonimización/pseudonimización y políticas de retención limitadas.
- En APIs de scoring, evitar loggear cuerpos de petición completos y restringir el acceso a datos históricos a propósitos de análisis/ML claramente definidos.

## Refresh Strategy
- Periodically scrape/ingest updated VGChartz / open sources and recompute features.
- Re-run drift/coverage checks on new titles and adjust class thresholds.
- Document dataset version (timestamp + source) in MLflow or metadata file.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `Gaming-Market-Intelligence/`
