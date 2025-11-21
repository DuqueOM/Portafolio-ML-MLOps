#!/bin/bash
# Script para ejecutar tests del portfolio TOP-3
set -e

echo "========================================="
echo "Running Tests for TOP-3 Projects"
echo "========================================="
echo ""

# Crear directorio de reports si no existe
mkdir -p reports

# Array de proyectos top-3
PROJECTS=(
    "BankChurn-Predictor"
    "CarVision-Market-Intelligence"
    "TelecomAI-Customer-Intelligence"
)

# Función para ejecutar tests de un proyecto
run_project_tests() {
    local project=$1
    echo ""
    echo "========================================="
    echo "Testing: $project"
    echo "========================================="
    
    cd "$project"
    
    # Activar venv si existe
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Instalar dependencias si requirements.txt existe
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install -q -r requirements.txt
    elif [ -f "requirements.in" ]; then
        echo "Installing dependencies from requirements.in..."
        pip install -q -r requirements.in
    fi
    
    # Ejecutar pytest con coverage
    echo "Running pytest..."
    if pytest --maxfail=1 --disable-warnings -q --cov=src 2>&1 | tee "../reports/${project}-pytest-log.txt"; then
        echo "✓ Tests passed for $project"
    else
        echo "✗ Tests failed for $project (see log for details)"
    fi
    
    # Generar coverage report
    echo "Generating coverage report..."
    coverage report -m 2>&1 | tee "../reports/${project}-coverage-report.txt"
    
    # Extraer coverage percentage
    COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}')
    echo "$project,$COVERAGE" >> "../reports/coverage-summary-TOP3.csv"
    
    cd ..
}

# Inicializar CSV de coverage
echo "Project,Coverage" > reports/coverage-summary-TOP3.csv

# Ejecutar tests para cada proyecto
for project in "${PROJECTS[@]}"; do
    if [ -d "$project" ]; then
        run_project_tests "$project"
    else
        echo "⚠ Warning: $project directory not found"
    fi
done

echo ""
echo "========================================="
echo "Test Execution Complete"
echo "========================================="
echo ""
echo "Reports saved in reports/ directory:"
ls -lh reports/*-pytest-log.txt reports/*-coverage-report.txt reports/coverage-summary-TOP3.csv
