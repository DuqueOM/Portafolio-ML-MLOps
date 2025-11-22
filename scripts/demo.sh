#!/bin/bash

# Demo Script - ML-MLOps Portfolio
# Levanta demo completa y ejecuta requests de prueba
# Uso: bash scripts/demo.sh

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ML-MLOps Portfolio - Demo Automatizado${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}\n"

# 1. Verificar Docker
echo -e "${BLUE}[1/6]${NC} Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker no está instalado${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose no está instalado${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker y docker-compose disponibles${NC}\n"

# 2. Limpiar contenedores previos
echo -e "${BLUE}[2/6]${NC} Limpiando contenedores previos..."
docker-compose -f docker-compose.demo.yml down --remove-orphans 2>/dev/null || true
echo -e "${GREEN}✓ Limpieza completada${NC}\n"

# 3. Construir imágenes
echo -e "${BLUE}[3/6]${NC} Construyendo imágenes Docker..."
echo -e "${YELLOW}Esto puede tomar varios minutos...${NC}"
docker-compose -f docker-compose.demo.yml build --parallel
echo -e "${GREEN}✓ Imágenes construidas${NC}\n"

# 4. Iniciar servicios
echo -e "${BLUE}[4/6]${NC} Iniciando servicios..."
docker-compose -f docker-compose.demo.yml up -d

# Esperar a que los servicios estén healthy
echo -e "${YELLOW}Esperando a que los servicios estén listos...${NC}"
sleep 10

for i in {1..30}; do
    if docker-compose -f docker-compose.demo.yml ps | grep -q "healthy"; then
        break
    fi
    echo -n "."
    sleep 2
done
echo ""
echo -e "${GREEN}✓ Servicios iniciados${NC}\n"

# 5. Mostrar servicios disponibles
echo -e "${BLUE}[5/6]${NC} Servicios disponibles:"
echo -e "  ${GREEN}►${NC} MLflow UI:          ${YELLOW}http://localhost:5000${NC}"
echo -e "  ${GREEN}►${NC} BankChurn API:      ${YELLOW}http://localhost:8001${NC}"
echo -e "  ${GREEN}►${NC} CarVision API:      ${YELLOW}http://localhost:8002${NC}"
echo -e "  ${GREEN}►${NC} CarVision Dashboard:${YELLOW}http://localhost:8501${NC}"
echo -e "  ${GREEN}►${NC} Telecom API:        ${YELLOW}http://localhost:8003${NC}\n"

# 6. Ejecutar requests de prueba
echo -e "${BLUE}[6/6]${NC} Ejecutando requests de prueba...\n"

# Health checks
echo -e "${YELLOW}► Testing BankChurn API health...${NC}"
curl -s http://localhost:8001/health | jq '.' 2>/dev/null || curl -s http://localhost:8001/health
echo ""

echo -e "${YELLOW}► Testing CarVision API health...${NC}"
curl -s http://localhost:8002/health | jq '.' 2>/dev/null || curl -s http://localhost:8002/health
echo ""

echo -e "${YELLOW}► Testing Telecom API health...${NC}"
curl -s http://localhost:8003/health | jq '.' 2>/dev/null || curl -s http://localhost:8003/health
echo ""

# Prediction example - BankChurn
echo -e "\n${YELLOW}► Testing BankChurn prediction...${NC}"
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "CreditScore": 650,
       "Geography": "France",
       "Gender": "Female",
       "Age": 40,
       "Tenure": 3,
       "Balance": 60000,
       "NumOfProducts": 2,
       "HasCrCard": 1,
       "IsActiveMember": 1,
       "EstimatedSalary": 50000
     }' | jq '.' 2>/dev/null || echo "Prediction executed (jq not available for formatting)"

echo ""

# Resumen final
echo -e "\n${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Demo iniciado exitosamente${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}Comandos útiles:${NC}"
echo -e "  ${BLUE}Ver logs:${NC}        docker-compose -f docker-compose.demo.yml logs -f"
echo -e "  ${BLUE}Detener demo:${NC}    docker-compose -f docker-compose.demo.yml down"
echo -e "  ${BLUE}Reiniciar:${NC}       docker-compose -f docker-compose.demo.yml restart"
echo ""

echo -e "${YELLOW}Próximos pasos:${NC}"
echo -e "  1. Abre ${YELLOW}http://localhost:5000${NC} para ver MLflow UI"
echo -e "  2. Abre ${YELLOW}http://localhost:8501${NC} para CarVision Dashboard"
echo -e "  3. Usa los endpoints de las APIs para hacer predicciones"
echo ""
