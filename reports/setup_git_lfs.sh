#!/bin/bash
# Script para configurar Git LFS para modelos grandes
# y archivos binarios en el portafolio

set -e

PORTFOLIO_ROOT="/home/duque_om/projects/Projects Tripe Ten"

echo "========================================="
echo "Git LFS Setup - Portfolio ML/MLOps"
echo "========================================="
echo ""

cd "$PORTFOLIO_ROOT"

# 1. Verificar instalación de Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo "[ERROR] Git LFS no está instalado"
    echo "Instalar con:"
    echo "  - Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  - macOS: brew install git-lfs"
    exit 1
fi

echo "[✓] Git LFS está instalado: $(git lfs version)"
echo ""

# 2. Inicializar Git LFS
if ! git lfs env &> /dev/null; then
    echo "[*] Instalando Git LFS hooks..."
    git lfs install
    echo "[✓] Git LFS inicializado"
else
    echo "[✓] Git LFS ya está configurado"
fi
echo ""

# 3. Configurar .gitattributes para archivos grandes
echo "[*] Configurando .gitattributes para modelos y archivos binarios..."

cat > .gitattributes << 'EOF'
# Git LFS Configuration - Portfolio ML/MLOps
# https://git-lfs.github.com/

# Model files (PyTorch, TensorFlow, Scikit-learn serialized)
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.sav filter=lfs diff=lfs merge=lfs -text

# Large model artifacts
models/**/*.pkl filter=lfs diff=lfs merge=lfs -text
models/**/*.joblib filter=lfs diff=lfs merge=lfs -text
models/**/*.h5 filter=lfs diff=lfs merge=lfs -text

# Binary data files (>100MB typically)
*.parquet filter=lfs diff=lfs merge=lfs -text
*.feather filter=lfs diff=lfs merge=lfs -text

# Compressed archives with models
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text

# Database files
*.db filter=lfs diff=lfs merge=lfs -text
*.sqlite filter=lfs diff=lfs merge=lfs -text

# Exclude virtual environments and cache from LFS
.venv/** -filter -diff -merge -text
__pycache__/** -filter -diff -merge -text
*.pyc -filter -diff -merge -text
EOF

git add .gitattributes
echo "[✓] .gitattributes configurado"
echo ""

# 4. Escanear archivos existentes que deberían estar en LFS
echo "[*] Escaneando archivos que deberían estar en LFS..."
echo ""

# Buscar modelos serializados
MODEL_FILES=$(find . -type f \( -name "*.pkl" -o -name "*.joblib" -o -name "*.h5" \) \
    ! -path "./.venv/*" ! -path "./venv/*" ! -path "*/__pycache__/*" 2>/dev/null || true)

if [ -n "$MODEL_FILES" ]; then
    echo "Archivos de modelos encontrados:"
    echo "$MODEL_FILES" | while read file; do
        SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
        echo "  - $file ($SIZE)"
    done
    echo ""
    
    read -p "¿Migrar estos archivos a Git LFS? [y/N]: " MIGRATE
    
    if [ "$MIGRATE" == "y" ] || [ "$MIGRATE" == "Y" ]; then
        echo "[*] Migrando archivos a Git LFS..."
        git lfs migrate import --include="*.pkl,*.joblib,*.h5" --everything
        echo "[✓] Archivos migrados a LFS"
    else
        echo "[!] Archivos NO migrados. Migrar manualmente si es necesario."
    fi
else
    echo "[!] No se encontraron archivos de modelos para migrar"
fi
echo ""

# 5. Mostrar status de LFS
echo "[*] Estado de Git LFS:"
git lfs ls-files
echo ""

# 6. Configurar límites de LFS (opcional)
echo "[*] Configurando límites de LFS..."
# Evitar tracking automático de archivos pequeños
git config lfs.fetchrecentalways false
git config lfs.fetchrecentrefsdays 7
echo "[✓] Límites configurados"
echo ""

# 7. Resumen
echo "========================================="
echo "Git LFS Setup Completado"
echo "========================================="
echo ""
echo "Archivos trackeados por LFS:"
git lfs ls-files | wc -l || echo "0"
echo ""
echo "Próximos pasos:"
echo "1. Commit .gitattributes: git commit -m 'Configure Git LFS'"
echo "2. Push con LFS: git push"
echo "3. Otros clones: git lfs pull"
echo ""
echo "Comandos útiles:"
echo "  - Ver archivos LFS: git lfs ls-files"
echo "  - Ver status: git lfs status"
echo "  - Prune old versions: git lfs prune"
echo ""
echo "Para más info: https://git-lfs.github.com/"
