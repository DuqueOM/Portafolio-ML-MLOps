# 📊 Plan de Mejora de Coverage - Análisis y Acción

**Fecha**: 2025-11-21  
**Objetivo**: Alcanzar >75% coverage en todos los proyectos

---

## 🎯 Situación Actual

| Proyecto | Coverage Actual | Target | Gap | Prioridad |
|----------|----------------|--------|-----|-----------|
| **TelecomAI** | 87% | 75% | +12% | ✅ **OK** |
| **CarVision** | 81% | 75% | +6% | ✅ **OK** |
| **OilWell** | 57% | 75% | -18% | 🔴 Alta |
| **Chicago** | 56% | 75% | -19% | 🔴 Alta |
| **BankChurn** (Tier-1) | 45% | 85% | -40% | 🔴 **Crítica** |
| **Gaming** | 39% | 75% | -36% | 🔴 Alta |
| **GoldRecovery** | 36% | 75% | -39% | 🔴 Alta |

**Proyectos OK**: 2/7 (29%)  
**Proyectos que necesitan mejora**: 5/7 (71%)

---

## 🔍 Análisis Detallado por Proyecto

### BankChurn-Predictor (45% → 85%)

**Módulos sin tests**:
- `cli.py`: 0% (115 líneas) - CLI no testeada
- `evaluation.py`: 0% (83 líneas) - Evaluación sin tests
- `prediction.py`: 0% (62 líneas) - Predicción sin tests
- `training.py`: 0% (112 líneas) - Training sin tests

**Acción tomada**:
- ✅ Creados test_training.py (18 tests)
- ✅ Creados test_evaluation.py (24 tests)
- ✅ Creados test_prediction.py (26 tests)

**Problema actual**:
- Tests creados tienen incompatibilidades con interfaces reales
- Necesitan ajuste para match con implementación actual

**Solución pragmática**:
1. Revisar interfaces reales de los módulos
2. Ajustar tests para match con las implementaciones
3. Enfocarse en tests de integración funcionales
4. Priorizar happy paths y casos críticos

---

### GoldRecovery-Process-Optimizer (36% → 75%)

**Módulos sin tests**:
- `main.py`: 28% coverage (302 líneas, 218 miss)
- `evaluate.py`: 0% (61 líneas)
- `app/`: 0% en todos los módulos

**Tests existentes**: Básicos (data, model, preprocessing)

**Estrategia**:
1. Testear funciones principales en `main.py`:
   - `ProcessDataLoader` class
   - `MetallurgicalPredictor` class
   - Funciones de entrenamiento
2. Testear `evaluate.py`:
   - Función `evaluate()`
   - Bootstrap MAE
   - Métricas sMAPE
3. Tests de integración para `app/`:
   - FastAPI endpoints básicos
   - Streamlit dashboard (smoke tests)

---

### Gaming-Market-Intelligence (39% → 75%)

**Módulos sin tests**:
- `main.py`: 0% (138 líneas)
- `evaluate.py`: 0% (31 líneas)
- `app/`: 0% en todos

**Estrategia similar a GoldRecovery**:
- Tests de funciones principales
- Tests de evaluación
- Tests básicos de API

---

### Chicago-Mobility-Analytics (56% → 75%)

**Gap menor**: Solo necesita +19%

**Estrategia**:
- Identificar módulos parcialmente testeados
- Agregar tests para funciones faltantes
- Tests de integración

---

### OilWell-Location-Optimizer (57% → 75%)

**Gap menor**: Solo necesita +18%

**Estrategia similar a Chicago**

---

## 🚀 Plan de Acción Pragmático

### Fase 1: Fix BankChurn (Crítico) - 2-3 horas

1. **Revisar interfaces reales**:
   ```bash
   cd BankChurn-Predictor
   # Revisar src/bankchurn/training.py
   # Revisar src/bankchurn/evaluation.py  
   # Revisar src/bankchurn/prediction.py
   ```

2. **Ajustar tests existentes**:
   - Corregir test_training.py para match con ChurnTrainer API real
   - Corregir test_evaluation.py para match con ModelEvaluator API real
   - Corregir test_prediction.py para match con ChurnPredictor API real

3. **Tests de integración**:
   - Test completo de pipeline: load → train → evaluate → predict
   - Test de serialización/deserialización
   - Tests con datos reales (muestras pequeñas)

4. **Target**: Alcanzar 75-80% coverage mínimo

### Fase 2: Proyectos con 36-39% (Alta) - 3-4 horas

**GoldRecovery y Gaming**: Coverage muy bajo

1. **Crear tests para main.py**:
   - Tests de clases principales
   - Tests de funciones de entrenamiento
   - Tests de métodos públicos

2. **Crear tests para evaluate.py**:
   - Tests de métricas
   - Tests de bootstrap
   - Tests de sMAPE

3. **Tests mínimos de app/**:
   - Smoke tests para FastAPI (endpoints responden)
   - Smoke tests para Streamlit (app carga)

4. **Target**: Alcanzar 75% coverage

### Fase 3: Proyectos con 56-57% (Media) - 1-2 horas

**Chicago y OilWell**: Solo necesitan boost pequeño

1. **Identificar gaps**:
   ```bash
   coverage report -m | grep -E "(Chicago|OilWell)"
   ```

2. **Agregar tests faltantes**:
   - Completar tests de funciones parcialmente cubiertas
   - Agregar tests de edge cases

3. **Target**: Alcanzar 75%+ coverage

---

## 📝 Checklist de Implementación

### BankChurn-Predictor
- [ ] Revisar API real de training.py
- [ ] Revisar API real de evaluation.py
- [ ] Revisar API real de prediction.py
- [ ] Ajustar test_training.py
- [ ] Ajustar test_evaluation.py
- [ ] Ajustar test_prediction.py
- [ ] Ejecutar tests y verificar coverage
- [ ] Crear tests de integración E2E
- [ ] Alcanzar 80%+ coverage

### GoldRecovery-Process-Optimizer
- [ ] Crear tests/test_main_core.py
- [ ] Crear tests/test_evaluate.py
- [ ] Crear tests/test_app_basic.py
- [ ] Ejecutar y verificar 75%+ coverage

### Gaming-Market-Intelligence
- [ ] Crear tests/test_main.py
- [ ] Crear tests/test_evaluate.py
- [ ] Crear tests/test_app_endpoints.py
- [ ] Verificar 75%+ coverage

### Chicago-Mobility-Analytics
- [ ] Revisar coverage report detallado
- [ ] Identificar funciones sin tests
- [ ] Agregar tests faltantes
- [ ] Verificar 75%+ coverage

### OilWell-Location-Optimizer
- [ ] Revisar coverage report detallado
- [ ] Identificar funciones sin tests
- [ ] Agregar tests faltantes
- [ ] Verificar 75%+ coverage

---

## 🛠️ Estrategia de Tests Pragmática

### Priorizar Tests de Alto Impacto

1. **Happy Path Tests** (70% del coverage):
   - Funciones principales con inputs válidos
   - Flujos normales de ejecución
   - Casos de uso comunes

2. **Error Handling Tests** (20%):
   - Inputs inválidos
   - Excepciones esperadas
   - Edge cases críticos

3. **Integration Tests** (10%):
   - Pipelines completos
   - Interacción entre módulos

### No Perder Tiempo en:

- Tests de código generado automáticamente
- Tests de librerías externas (ya testeadas)
- Tests de UI interactiva (Streamlit dashboard visual)
- Tests de configuración trivial

### Enfoque en:

- **Lógica de negocio**: Funciones que implementan algoritmos
- **Transformación de datos**: Preprocesamiento, feature engineering
- **Modelos**: Training, evaluation, prediction
- **APIs**: Endpoints críticos

---

## 📊 Métricas de Éxito

### Por Proyecto

- [ ] BankChurn: 80%+ coverage (desde 45%)
- [ ] GoldRecovery: 75%+ coverage (desde 36%)
- [ ] Gaming: 75%+ coverage (desde 39%)
- [ ] Chicago: 75%+ coverage (desde 56%)
- [ ] OilWell: 75%+ coverage (desde 57%)
- [ ] CarVision: Mantener 81%+ ✅
- [ ] TelecomAI: Mantener 87%+ ✅

### Global

- [ ] **Promedio**: >75% coverage
- [ ] **Mínimo**: Todos los proyectos >70%
- [ ] **Tier-1 (BankChurn)**: >80%

---

## 🎯 Timeline Estimado

- **Fase 1 (BankChurn)**: 2-3 horas
- **Fase 2 (GoldRecovery + Gaming)**: 3-4 horas
- **Fase 3 (Chicago + OilWell)**: 1-2 horas

**Total estimado**: 6-9 horas de trabajo enfocado

---

## 📈 Próximos Pasos Inmediatos

1. **Ahora**: Fix tests de BankChurn
   ```bash
   cd BankChurn-Predictor
   # Revisar interfaces
   python -c "from src.bankchurn import training; help(training.ChurnTrainer)"
   python -c "from src.bankchurn import evaluation; help(evaluation.ModelEvaluator)"
   python -c "from src.bankchurn import prediction; help(prediction.ChurnPredictor)"
   ```

2. **Luego**: Ajustar tests según interfaces reales

3. **Después**: Ejecutar y validar coverage >80%

4. **Siguiente**: Repetir para otros proyectos

---

**Status**: 🟡 En progreso - Tests creados, necesitan ajustes  
**Prioridad**: 🔴 Alta - Coverage crítico para portfolio profesional
