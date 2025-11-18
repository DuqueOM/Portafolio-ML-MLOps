# Data Card — TelecomAI Customer Intelligence

## Dataset Overview
- **Name:** users_behavior.csv
- **Records:** ~3,214 subscribers
- **Features:** `calls`, `minutes`, `messages`, `mb_used`
- **Target:** `is_ultra` (1 = recomendar plan Ultra, 0 = plan Smart)

## Source & Licensing
- **Origin:** TripleTen / educational dataset (telecom usage simulation).
- **License:** MIT-friendly for portfolio/demo (ver `DATA_LICENSE`).
- **PII:** No contiene PII; datos agregados anonimizados.

## Schema Snapshot
| Column | Type | Notes |
|--------|------|-------|
| `calls` | float | Cantidad de llamadas al mes |
| `minutes` | float | Minutos totales usados |
| `messages` | int | Mensajes SMS |
| `mb_used` | float | Datos móviles en MB |
| `is_ultra` | int | Target binario |

## Splits & Versioning
- Split estándar 80/20 estratificado (configurable en `configs/config.yaml`, seed=42).
- Artefactos de entrenamiento guardados en `artifacts/` (`model.joblib`, `preprocessor.joblib`, `metrics.json`).
- Dataset crudo permanece en `users_behavior.csv`; procesado se genera en runtime.

## Data Quality Considerations
- Pequeñas variaciones de escala (minutes ≈ calls*avg_call_length) → normalizado con `StandardScaler`.
- Sin valores faltantes en la versión base.
- Distribución sintética; no refleja patrones reales de uso multi-dispositivo.

## Bias & Ethical Notes
- Ausencia de atributos demográficos → fairness difícil de auditar, pero también evita sesgos explícitos.
- Datos sintéticos no consideran estacionalidad ni cambios de red, por lo que las métricas son ilustrativas.

## Privacy & Targeting Ethics
- **Ausencia de PII:** el dataset no contiene identificadores personales ni datos sensibles; sólo patrones agregados de uso (llamadas, minutos, mensajes, datos).
- **Riesgos de targeting:** incluso sin PII, la recomendación de planes puede generar experiencias de upsell agresivo si se contacta demasiado a los clientes o se les ofrece sistemáticamente el plan más caro.
- **Uso recomendado:**
  - Utilizar las predicciones como insumo para campañas segmentadas con límites de frecuencia y reglas de elegibilidad claras.
  - Evitar penalizar a segmentos concretos de uso (p. ej., heavy data users) sin revisar impacto en satisfacción y churn.
  - No combinar este dataset con atributos sensibles externos sin un análisis formal de fairness y cumplimiento normativo.

## Refresh Strategy
- Reemplazar con datos reales (anónimos) antes de producción.
- Ejecutar `python monitoring/check_drift.py` ante cada refresco para comparar contra histórico.
- Registrar versión del CSV y fecha en README o MLflow.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `TelecomAI-Customer-Intelligence/`
