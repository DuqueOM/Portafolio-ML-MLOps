#!/bin/bash
# Script para configurar DVC en el portafolio
# Incluye configuración de remote storage y tracking de datasets

set -e

PORTFOLIO_ROOT="/home/duque_om/projects/Projects Tripe Ten"

echo "========================================="
echo "DVC Setup - Portfolio ML/MLOps"
echo "========================================="
echo ""

cd "$PORTFOLIO_ROOT"

# 1. Verificar instalación de DVC
if ! command -v dvc &> /dev/null; then
    echo "[ERROR] DVC no está instalado"
    echo "Instalar con: pip install 'dvc[s3]'"
    exit 1
fi

echo "[✓] DVC está instalado: $(dvc version)"
echo ""

# 2. Inicializar DVC
if [ ! -d ".dvc" ]; then
    echo "[*] Inicializando DVC..."
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC"
    echo "[✓] DVC inicializado"
else
    echo "[✓] DVC ya está inicializado"
fi
echo ""

# 3. Configurar remote storage (ejemplo con S3)
echo "[*] Configurando remote storage..."
echo ""
echo "Opciones de remote storage:"
echo "1. S3 (AWS)"
echo "2. Local (para testing)"
echo "3. Google Drive"
echo "4. Azure Blob Storage"
echo ""
read -p "Seleccionar opción [1-4]: " STORAGE_OPTION

case $STORAGE_OPTION in
    1)
        echo "[*] Configurando S3 remote..."
        read -p "Bucket S3 (ej: s3://my-ml-bucket/datasets): " S3_BUCKET
        dvc remote add -d storage "$S3_BUCKET"
        echo ""
        echo "[!] Configurar credenciales AWS:"
        echo "    aws configure"
        echo "    o establecer AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY"
        ;;
    2)
        echo "[*] Configurando almacenamiento local..."
        mkdir -p "$PORTFOLIO_ROOT/.dvc-storage"
        dvc remote add -d storage "$PORTFOLIO_ROOT/.dvc-storage"
        echo "[✓] Local storage configurado en: .dvc-storage/"
        ;;
    3)
        echo "[*] Configurando Google Drive remote..."
        read -p "ID de carpeta de Google Drive: " GDRIVE_ID
        dvc remote add -d storage "gdrive://$GDRIVE_ID"
        echo ""
        echo "[!] Autorizar DVC con Google Drive:"
        echo "    dvc remote modify storage gdrive_acknowledge_abuse true"
        ;;
    4)
        echo "[*] Configurando Azure Blob Storage..."
        read -p "Container Azure (ej: azure://container/path): " AZURE_PATH
        dvc remote add -d storage "$AZURE_PATH"
        ;;
    *)
        echo "[ERROR] Opción inválida"
        exit 1
        ;;
esac

git add .dvc/config
git commit -m "Configure DVC remote storage"
echo "[✓] Remote storage configurado"
echo ""

# 4. Escanear datasets grandes
echo "[*] Escaneando datasets grandes (>10MB)..."
echo ""

# Buscar archivos CSV grandes
CSV_FILES=$(find . -name "*.csv" -size +10M 2>/dev/null | grep -v ".venv" | grep -v "node_modules" || true)

if [ -n "$CSV_FILES" ]; then
    echo "Datasets CSV encontrados:"
    echo "$CSV_FILES" | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "  - $file ($SIZE)"
    done
    echo ""
    read -p "¿Trackear estos archivos con DVC? [y/N]: " TRACK_CSV
    
    if [ "$TRACK_CSV" == "y" ] || [ "$TRACK_CSV" == "Y" ]; then
        echo "$CSV_FILES" | while read file; do
            echo "[*] Tracking: $file"
            dvc add "$file"
            git add "${file}.dvc" ".gitignore"
        done
        git commit -m "Track large datasets with DVC"
        echo "[✓] Datasets trackeados"
    fi
else
    echo "[!] No se encontraron datasets CSV >10MB"
fi
echo ""

# 5. Crear data/README.md con checksums
echo "[*] Creando data/README.md..."

cat > "$PORTFOLIO_ROOT/data/README.md" << 'EOF'
# Datasets - Portfolio ML/MLOps

Este directorio contiene los datasets utilizados en el portafolio.

## Gestión con DVC

Los datasets están versionados con DVC (Data Version Control).

### Descargar datasets

```bash
# Descargar todos los datos
dvc pull

# Descargar un dataset específico
dvc pull data/specific_dataset.csv.dvc
```

### Actualizar datasets

```bash
# Modificar archivo
# ...

# Re-trackear cambios
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Update dataset"

# Subir nueva versión
dvc push
```

## Datasets por Proyecto

### BankChurn-Predictor
- **Archivo**: `bankchurn/Churn_Modelling.csv`
- **Tamaño**: ~2MB
- **Registros**: 10,000
- **Features**: 14

### CarVision-Market-Intelligence
- **Archivo**: `carvision/vehicles_us.csv`
- **Tamaño**: ~50MB
- **Registros**: 51,525
- **Features**: 13

### TelecomAI-Customer-Intelligence
- **Archivo**: `telecom/telecom_users.csv`
- **Tamaño**: ~5MB
- **Registros**: 7,043
- **Features**: 20

### Chicago-Mobility-Analytics
- **Archivo**: `chicago/taxi_data.csv`
- **Tamaño**: ~30MB
- **Registros**: Variable
- **Features**: 4-5

### GoldRecovery-Process-Optimizer
- **Archivo**: `gold/gold_recovery_full.csv`
- **Tamaño**: ~20MB
- **Registros**: 16,860
- **Features**: 87

### Gaming-Market-Intelligence
- **Archivo**: `gaming/games.csv`
- **Tamaño**: ~5MB
- **Registros**: 16,715
- **Features**: 11

### OilWell-Location-Optimizer
- **Archivos**: `oil/geo_data_[0-2].csv`
- **Tamaño**: ~1MB cada uno
- **Registros**: 100,000 por archivo
- **Features**: 5

## Checksums y Validación

DVC genera checksums automáticamente en archivos `.dvc`.

Para validar integridad:

```bash
dvc status
```

## Licencias y Atribución

Los datasets son educativos/sintéticos proporcionados por TripleTen.
No contienen datos sensibles reales.

Ver `LICENSE` en cada directorio de proyecto para detalles.
EOF

mkdir -p "$PORTFOLIO_ROOT/data"
git add "$PORTFOLIO_ROOT/data/README.md"
git commit -m "Add data README with DVC documentation"

echo "[✓] data/README.md creado"
echo ""

# 6. Resumen
echo "========================================="
echo "DVC Setup Completado"
echo "========================================="
echo ""
echo "Próximos pasos:"
echo "1. Verificar remote: dvc remote list"
echo "2. Push datos: dvc push"
echo "3. En otros clones: dvc pull"
echo ""
echo "Para más info: https://dvc.org/doc"
