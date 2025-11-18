# Data Card — OilWell Location Optimizer

## Dataset Overview
- **Name:** geo_data_{0,1,2}.csv (synthetic geological samples per region)
- **Records:** ~100,000 rows por región (feature vectors + target `product`).
- **Features:** `f0`, `f1`, `f2` (normalized logging features) y `product` (yield estimado).
- **Target:** `product` — producción esperada para entrenar modelos de selección de pozos.

## Source & Licensing
- **Origin:** TripleTen educational dataset (Oil Well Optimization sprint).
- **License:** Uso educativo/portafolio (ver `DATA_LICENSE`).
- **PII:** Ninguna. Datos completamente sintéticos.

## Schema Snapshot
| Column | Type | Notes |
|--------|------|-------|
| `id` | string | Identificador único de pozo |
| `f0`, `f1`, `f2` | float | Features numéricas simuladas |
| `product` | float | Target (producción esperada en miles de barriles) |

## Splits & Versioning
- Dataset se divide por región (0, 1, 2).
- En entrenamiento, se realizan splits 75/25 por región (seed=42) antes de ajustar `LinearRegression`.
- Resultados/artefactos se guardan bajo `artifacts/` (modelos, métricas, bootstrap, Monte Carlo, optimizaciones).

## Quality Considerations
- Datos sin ruido contextually real → no incluyen incertidumbre geológica real.
- Cada región tiene diferente distribución; mezclar sin normalizar puede introducir drift.
- No existen valores faltantes en los CSV originales; se revisa deduplicación por `id` antes de modelar.

## Bias & Ethical Notes
- Dataset sintético no captura impactos sociales/ambientales reales.
- Supuestos de CAPEX y revenue en scripts son constantes; pueden inducir sesgos financieros.
- Modelo no considera permisos, comunidades ni riesgos ambientales (documentados en `docs/assumptions.md`).

## Privacy & Data Governance

- Aunque los datos sean sintéticos en este proyecto, en contextos reales la información geológica/financiera puede ser altamente sensible.
- Se recomienda anonimizar identificadores de pozos y contratos, y limitar el uso de datos a equipos autorizados (riesgo, planeación, geología).
- Evitar compartir datasets completos fuera de la organización sin procesos formales de anonimización y acuerdos de confidencialidad.

## Refresh Strategy
- Sustituir con mediciones reales o simulaciones actualizadas antes de despliegues productivos.
- Recalibrar parámetros de CAPEX, revenue, constraints cuando cambien precios del crudo o políticas.
- Versionar datasets con DVC/MLflow cuando se integren fuentes reales.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `OilWell-Location-Optimizer/`
