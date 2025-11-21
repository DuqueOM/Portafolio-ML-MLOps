#!/bin/bash
# Script para ejecutar tests en todos los proyectos del portafolio
# y generar reportes de coverage

set -e  # Exit on error

PORTFOLIO_ROOT="/home/duque_om/projects/Projects Tripe Ten"
REPORTS_DIR="$PORTFOLIO_ROOT/reports"

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Proyectos a testear
PROJECTS=(
    "BankChurn-Predictor"
    "CarVision-Market-Intelligence"
    "TelecomAI-Customer-Intelligence"
    "Chicago-Mobility-Analytics"
    "GoldRecovery-Process-Optimizer"
    "Gaming-Market-Intelligence"
    "OilWell-Location-Optimizer"
)

echo "========================================="
echo "Portfolio ML/MLOps - Test Runner"
echo "========================================="
echo ""

# Función para ejecutar tests en un proyecto
run_project_tests() {
    local project=$1
    local project_dir="$PORTFOLIO_ROOT/$project"
    
    echo -e "${YELLOW}[*] Procesando: $project${NC}"
    
    if [ ! -d "$project_dir" ]; then
        echo -e "${RED}[ERROR] Directorio no encontrado: $project_dir${NC}"
        return 1
    fi
    
    cd "$project_dir"
    
    # Verificar si existe requirements.txt
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}[ERROR] No se encontró requirements.txt en $project${NC}"
        return 1
    fi
    
    # Crear venv si no existe
    if [ ! -d ".venv" ]; then
        echo "[*] Creando entorno virtual..."
        python3 -m venv .venv
    fi
    
    # Activar venv e instalar dependencias
    source .venv/bin/activate
    
    echo "[*] Instalando dependencias..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > "$REPORTS_DIR/${project}-install.log" 2>&1
    
    # Ejecutar tests con pytest
    echo "[*] Ejecutando tests..."
    if [ -d "tests" ]; then
        pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing \
            2>&1 | tee "$REPORTS_DIR/${project}-pytest.txt"
        
        # Generar reporte de coverage detallado
        coverage report -m > "$REPORTS_DIR/${project}-coverage.txt" 2>&1
        
        # Extraer porcentaje de coverage
        COVERAGE=$(coverage report | tail -1 | awk '{print $NF}')
        echo -e "${GREEN}[✓] Tests completados. Coverage: $COVERAGE${NC}"
        echo "$project,$COVERAGE" >> "$REPORTS_DIR/coverage-summary.csv"
    else
        echo -e "${YELLOW}[!] No se encontró directorio tests/${NC}"
    fi
    
    deactivate
    cd "$PORTFOLIO_ROOT"
    echo ""
}

# Crear archivo CSV para resumen
echo "Project,Coverage" > "$REPORTS_DIR/coverage-summary.csv"

# Ejecutar tests en cada proyecto
for project in "${PROJECTS[@]}"; do
    run_project_tests "$project" || echo -e "${RED}[ERROR] Falló: $project${NC}"
done

echo "========================================="
echo "Resumen de Coverage"
echo "========================================="
cat "$REPORTS_DIR/coverage-summary.csv"
echo ""

echo -e "${GREEN}[✓] Tests completados. Reportes en: $REPORTS_DIR${NC}"
