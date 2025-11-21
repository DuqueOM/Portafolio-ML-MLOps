#!/bin/bash
# Script para instalar herramientas de seguridad necesarias
# Gitleaks, Trivy, DVC, Git LFS

set -e

echo "========================================="
echo "Instalación de Herramientas de Seguridad"
echo "========================================="
echo ""

# Detectar sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    echo "Sistema operativo no soportado: $OSTYPE"
    exit 1
fi

# Instalar Gitleaks
echo "[*] Instalando Gitleaks..."
if ! command -v gitleaks &> /dev/null; then
    if [ "$OS" == "linux" ]; then
        wget -qO- https://github.com/gitleaks/gitleaks/releases/download/v8.18.0/gitleaks_8.18.0_linux_x64.tar.gz | \
            sudo tar -xz -C /usr/local/bin gitleaks
    elif [ "$OS" == "mac" ]; then
        brew install gitleaks
    fi
    echo "[✓] Gitleaks instalado"
else
    echo "[✓] Gitleaks ya está instalado"
fi

# Instalar Trivy
echo "[*] Instalando Trivy..."
if ! command -v trivy &> /dev/null; then
    if [ "$OS" == "linux" ]; then
        wget -qO- https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | \
            sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy -y
    elif [ "$OS" == "mac" ]; then
        brew install aquasecurity/trivy/trivy
    fi
    echo "[✓] Trivy instalado"
else
    echo "[✓] Trivy ya está instalado"
fi

# Instalar DVC
echo "[*] Instalando DVC..."
if ! command -v dvc &> /dev/null; then
    pip3 install 'dvc[s3]'
    echo "[✓] DVC instalado"
else
    echo "[✓] DVC ya está instalado"
fi

# Instalar Git LFS
echo "[*] Instalando Git LFS..."
if ! command -v git-lfs &> /dev/null; then
    if [ "$OS" == "linux" ]; then
        sudo apt-get install git-lfs -y
    elif [ "$OS" == "mac" ]; then
        brew install git-lfs
    fi
    git lfs install
    echo "[✓] Git LFS instalado"
else
    echo "[✓] Git LFS ya está instalado"
fi

echo ""
echo "========================================="
echo "Verificación de Instalación"
echo "========================================="
echo "Gitleaks: $(gitleaks version 2>&1 || echo 'ERROR')"
echo "Trivy: $(trivy --version 2>&1 | head -1 || echo 'ERROR')"
echo "DVC: $(dvc version || echo 'ERROR')"
echo "Git LFS: $(git lfs version || echo 'ERROR')"
echo ""
echo "[✓] Instalación completada"
