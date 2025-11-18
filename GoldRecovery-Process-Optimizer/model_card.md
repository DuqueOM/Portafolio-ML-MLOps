# Model Card: GoldRecovery Process Optimizer

## Model Details
- **Model type:** Ensemble regressor (XGBoost + LightGBM + RandomForest)
- **Task:** Regression of `final.output.recovery` (0–100%)
- **Version:** 1.0.0
- **Author:** Daniel Duque

## Intended Use
- **Primary purpose:** Predecir y optimizar la recuperación final de oro en circuitos de flotación/limpieza.
- **Users:** Ingenieros de procesos, científicos de datos, operadores de planta.
- **Scope:** Asistencia a toma de decisiones, no reemplaza control legal/regulatorio ni responsabilidad operativa.

## Training Data
- **Origen:** CSVs `gold_recovery_train.csv`, `gold_recovery_test.csv`, `gold_recovery_full.csv`.
- **Features:** Variables de proceso, concentraciones (Au/Ag/Pb) por etapa, estados de celdas, insumos, tiempos.
- **Preprocesamiento:** Limpieza de nulos, cálculo de ratios, features temporales; imputación por mediana.

## Evaluation Data & Metrics
- **Conjunto:** `gold_recovery_test.csv` (hold-out)
- **Métricas:** sMAPE, MAE, RMSE, R², MAE baseline (Dummy)
- **Intervalos:** IC95 de MAE vía bootstrap (configurable)

### Impacto operativo (ejemplo ilustrativo)
- Supongamos que el baseline (promedio) obtiene sMAPE ≈ 15% y MAE ≈ 3.0 puntos de recuperación,
  mientras que el ensemble logra sMAPE ≈ 10% y MAE ≈ 2.0 en el mismo conjunto de test.
- En un circuito con 100 toneladas/hora de concentrado, una reducción de 1 punto de MAE en recuperación
  puede representar del orden de **1 tonelada/hora adicional equivalente** dentro de los rangos operativos.
- Menor error y variabilidad en `final.output.recovery` facilita:
  - operar más cerca de los límites de recuperación objetivo sin sobrepasar restricciones de proceso,
  - reducir retrabajos y ajustes manuales,
  - mejorar la estabilidad del circuito y la predictibilidad de producción.

## Ethical Considerations & Risks
- Posibles sesgos por condiciones específicas de una planta y época; pobre generalización a otras minas.
- Riesgo de sobreconfianza en predicciones; debe existir supervisión humana.
- Cambios de distribución (drift) por variabilidad mineralógica afectan performance; se requiere monitoreo.
- Uso indebido: optimizaciones que comprometan seguridad, ambiente o calidad; deben existir límites/reglas del proceso.

## Limitations
- No modela explícitamente dinámica completa (no es un gemelo digital); predicción estática por lote/ventana.
- Datos faltantes y señales ruidosas pueden degradar performance; se recomienda QA.
- No incluye costos de reactivos/energía por defecto; requiere integración para optimización económica completa.

## Recommendations
- Validación por periodo y por campaña minera; pruebas A/B controladas.
- Monitoreo de drift (PSI, KS) y reentrenamiento periódico.
- Alertas cuando input está fuera del rango observado en entrenamiento.
- Registro de versiones del modelo/datos para auditoría.

## Environmental and Safety
- Garantizar que setpoints sugeridos respeten restricciones de proceso, seguridad, descarga y normativa.

## Privacy & Data Governance

- Este proyecto se basa en datos industriales anonimizados para fines educativos; no incluye identificadores directos de planta ni de campañas reales.
- En un despliegue real, se deben clasificar los datos de proceso como sensibles/confidenciales y establecer controles de acceso y retención adecuados.
- Evitar registrar en logs parámetros que puedan revelar detalles de operación, especialmente si el sistema se expone fuera de la red interna.

## Caveats
- Este modelo es material educativo/demostrativo. No use en producción sin validaciones y controles adecuados.
