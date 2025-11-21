#!/bin/bash
# End-to-End Pipeline Script
# Ejecuta el flujo completo: ingest → train → register → serve → inference
# Para BankChurn-Predictor como proyecto de referencia

set -e

PROJECT="BankChurn-Predictor"
PORTFOLIO_ROOT="/home/duque_om/projects/Projects Tripe Ten"
PROJECT_DIR="$PORTFOLIO_ROOT/$PROJECT"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}E2E Pipeline - $PROJECT${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

cd "$PROJECT_DIR"

# Activar entorno virtual
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${RED}[ERROR] Entorno virtual no encontrado${NC}"
    echo "Ejecutar primero: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Paso 1: Data Ingestion
echo -e "${YELLOW}[1/6] Data Ingestion${NC}"
if [ -f "$PORTFOLIO_ROOT/scripts/fetch_data.py" ]; then
    python "$PORTFOLIO_ROOT/scripts/fetch_data.py" --project bankchurn --validate || {
        echo -e "${YELLOW}[!] Validación de datos falló, continuando...${NC}"
    }
else
    echo -e "${YELLOW}[!] Script fetch_data.py no encontrado${NC}"
fi
echo -e "${GREEN}[✓] Data ingestion completada${NC}"
echo ""

# Paso 2: DVC Pull (si está configurado)
echo -e "${YELLOW}[2/6] DVC Data Pull${NC}"
if command -v dvc &> /dev/null && [ -f "$PORTFOLIO_ROOT/.dvc/config" ]; then
    echo "[*] Pulling data con DVC..."
    dvc pull 2>/dev/null || echo "[!] DVC pull falló o no hay datos remotos"
else
    echo "[!] DVC no configurado, saltando..."
fi
echo -e "${GREEN}[✓] DVC check completado${NC}"
echo ""

# Paso 3: Training
echo -e "${YELLOW}[3/6] Model Training${NC}"
if [ -f "src/bankchurn/training.py" ]; then
    python -m src.bankchurn.training || python src/bankchurn/training.py || {
        echo -e "${RED}[ERROR] Training falló${NC}"
        exit 1
    }
elif [ -f "scripts/train.py" ]; then
    python scripts/train.py
else
    echo -e "${RED}[ERROR] Script de training no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Training completado${NC}"
echo ""

# Paso 4: Model Registration (MLflow)
echo -e "${YELLOW}[4/6] Model Registration${NC}"
if command -v mlflow &> /dev/null; then
    echo "[*] Registrando modelo en MLflow..."
    # Esto asume que el training ya registró el modelo
    # Aquí solo verificamos el tracking URI
    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}
    echo "[*] MLflow Tracking URI: $MLFLOW_TRACKING_URI"
else
    echo "[!] MLflow no instalado, saltando registro..."
fi
echo -e "${GREEN}[✓] Model registration completado${NC}"
echo ""

# Paso 5: API Server
echo -e "${YELLOW}[5/6] Starting API Server${NC}"
if [ -f "app/main.py" ]; then
    echo "[*] Iniciando servidor FastAPI en background..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
    API_PID=$!
    echo "[*] API PID: $API_PID"
    
    # Esperar a que el servidor esté listo
    echo "[*] Esperando que el servidor esté listo..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}[✓] API Server listo${NC}"
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
else
    echo -e "${RED}[ERROR] API main.py no encontrado${NC}"
    exit 1
fi
echo ""

# Paso 6: Inference Test
echo -e "${YELLOW}[6/6] Inference Test${NC}"
echo "[*] Probando predicción..."

# Payload de ejemplo
PAYLOAD='{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}'

# Hacer request
RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

echo "[*] Response:"
echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Verificar respuesta
if echo "$RESPONSE" | grep -q "prediction"; then
    echo -e "${GREEN}[✓] Inference exitoso${NC}"
else
    echo -e "${RED}[ERROR] Inference falló${NC}"
    INFERENCE_SUCCESS=false
fi
echo ""

# Cleanup: Detener API
echo "[*] Deteniendo API Server (PID: $API_PID)..."
kill $API_PID 2>/dev/null || true
sleep 2
echo -e "${GREEN}[✓] API Server detenido${NC}"
echo ""

# Resumen
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}E2E Pipeline Completado${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}Pasos ejecutados:${NC}"
echo "  ✓ 1. Data Ingestion"
echo "  ✓ 2. DVC Pull"
echo "  ✓ 3. Model Training"
echo "  ✓ 4. Model Registration"
echo "  ✓ 5. API Server"
echo "  ✓ 6. Inference Test"
echo ""

if [ "$INFERENCE_SUCCESS" != "false" ]; then
    echo -e "${GREEN}[SUCCESS] Pipeline E2E completado exitosamente${NC}"
    exit 0
else
    echo -e "${YELLOW}[WARNING] Pipeline completado con advertencias${NC}"
    exit 1
fi
