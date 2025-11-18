# Model Card — OilWell Location Optimizer

- Model: LinearRegression (scikit-learn)
- Target: `product` (unidades de producción estimadas por pozo)
- Features: `f0`, `f1`, `f2`
- Regions: 0, 1, 2
- Version: 1.0.0

## Intended Use
- Selección de los 200 pozos por región con mayor producción esperada y análisis de riesgo vía bootstrap.
- No debe usarse para estimaciones de reservas certificadas ni decisiones financieras sin validación humana.

## Data
- Origen: Datasets sintéticos educativos del proyecto (TripleTen / curso). Columnas: `id`, `f0`, `f1`, `f2`, `product`.
- Preprocesamiento: deduplicación por `id` conservando mayor `product`; mezcla determinista.

## Performance
- Métrica primaria: RMSE en validación (75/25). Baseline: DummyRegressor (media).
- Evaluación de riesgo: distribución de utilidades por bootstrap (1000 iteraciones, 500/200) con pérdida = utilidad < 0.

### Runtime (demo, entorno de referencia)
- Entrenamiento completo (modelos lineales por 3 regiones sobre `geo_data_*.csv`) en CPU estándar: del orden de decenas de segundos.
- Evaluación de riesgo vía bootstrap (1,000 iteraciones por región) sobre las 3 regiones: del orden de pocos minutos en CPU (dependiente del hardware).
- Inferencia de API (`POST /predict`) sobre pocos registros: latencia típica <100 ms por request en entorno local.

## Limitations and Risks
- Supone linealidad entre features y `product`.
- Sensible a shift de distribución (drift) entre regiones y en el tiempo.
- Posible leakage si se altera el procedimiento de deduplicación.
- Supone costos e ingresos constantes: `revenue_per_unit` y `total_investment` fijos.

## Ethical Considerations
- Riesgo de sesgos si las features no representan condiciones geológicas reales.
- Impacto ambiental y social no modelado; el modelado numérico no reemplaza procesos regulatorios.

## Security and Privacy
- Datos no personales. No se maneja información sensible.

## Privacy & Data Governance

- Las `geo_data_*.csv` de este proyecto son sintéticas; en un entorno real deberían anonimizarse coordenadas/IDs de pozos y separar claramente datos técnicos de información contractual.
- Evitar registrar payloads completos de solicitudes de scoring que incluyan identificadores de activos o parámetros financieros sensibles.
- Definir políticas de acceso y retención de datos (quién puede ver qué, y por cuánto tiempo) antes de desplegar modelos similares en contextos productivos.

## Maintenance
- Versionar artefactos en `artifacts/` y registrar métricas en `artifacts/metrics.json` y `artifacts/risk_results.json`.
- Reentrenar si cambia el dataset, los supuestos financieros o las features.
