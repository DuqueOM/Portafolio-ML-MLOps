#!/bin/bash
# Script para ejecutar scans de seguridad completos
# Gitleaks + Trivy en todos los contenedores

set -e

PORTFOLIO_ROOT="/home/duque_om/projects/Projects Tripe Ten"
REPORTS_DIR="$PORTFOLIO_ROOT/reports"

echo "========================================="
echo "Security Scan - Portfolio ML/MLOps"
echo "========================================="
echo ""

cd "$PORTFOLIO_ROOT"

# 1. Gitleaks - Scan de secretos
echo "[*] Ejecutando Gitleaks (secret detection)..."
if command -v gitleaks &> /dev/null; then
    gitleaks detect \
        --source . \
        --report-path "$REPORTS_DIR/gitleaks-report.json" \
        --report-format json \
        --verbose 2>&1 | tee "$REPORTS_DIR/gitleaks-output.txt"
    
    # Generar reporte legible
    if [ -f "$REPORTS_DIR/gitleaks-report.json" ]; then
        SECRETS_COUNT=$(jq length "$REPORTS_DIR/gitleaks-report.json" 2>/dev/null || echo "0")
        echo ""
        echo "========================================="
        echo "Gitleaks Results"
        echo "========================================="
        echo "Secretos detectados: $SECRETS_COUNT"
        
        if [ "$SECRETS_COUNT" -gt 0 ]; then
            echo "[!] ALERTA: Se encontraron secretos en el repositorio"
            echo "Ver detalles en: $REPORTS_DIR/gitleaks-report.json"
        else
            echo "[✓] No se encontraron secretos"
        fi
    fi
else
    echo "[ERROR] Gitleaks no está instalado"
    echo "Ejecutar: bash $REPORTS_DIR/install_security_tools.sh"
fi
echo ""

# 2. Trivy - Scan de contenedores
echo "[*] Escaneando contenedores con Trivy..."
if command -v trivy &> /dev/null; then
    
    # Proyectos con Dockerfile
    PROJECTS_WITH_DOCKER=(
        "BankChurn-Predictor"
        "CarVision-Market-Intelligence"
        "TelecomAI-Customer-Intelligence"
    )
    
    for project in "${PROJECTS_WITH_DOCKER[@]}"; do
        PROJECT_DIR="$PORTFOLIO_ROOT/$project"
        
        if [ -f "$PROJECT_DIR/Dockerfile" ]; then
            echo "[*] Building y escaneando: $project"
            
            # Build imagen
            docker build -t "ml-portfolio-${project,,}:latest" "$PROJECT_DIR" \
                > "$REPORTS_DIR/${project}-docker-build.log" 2>&1 || {
                echo "[ERROR] Fallo el build de $project"
                continue
            }
            
            # Scan con Trivy
            trivy image \
                --severity HIGH,CRITICAL \
                --format table \
                "ml-portfolio-${project,,}:latest" \
                | tee "$REPORTS_DIR/${project}-trivy.txt"
            
            # Generar reporte JSON
            trivy image \
                --severity HIGH,CRITICAL \
                --format json \
                --output "$REPORTS_DIR/${project}-trivy.json" \
                "ml-portfolio-${project,,}:latest"
            
            echo "[✓] Scan completado: $project"
            echo ""
        fi
    done
    
else
    echo "[ERROR] Trivy no está instalado"
    echo "Ejecutar: bash $REPORTS_DIR/install_security_tools.sh"
fi

echo "========================================="
echo "Security Scan Completado"
echo "========================================="
echo "Reportes generados en: $REPORTS_DIR"
echo ""
echo "Archivos:"
echo "  - gitleaks-report.json"
echo "  - gitleaks-output.txt"
echo "  - *-trivy.txt (por proyecto)"
echo "  - *-trivy.json (por proyecto)"
echo ""
