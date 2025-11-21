#!/usr/bin/env python3
"""
Script para descargar/validar datasets del portafolio ML/MLOps.
Genera checksums y verifica integridad de datos.

Uso:
    python scripts/fetch_data.py --project bankchurn
    python scripts/fetch_data.py --project all --validate
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict

# Portfolio root
PORTFOLIO_ROOT = Path(__file__).parent.parent

# Dataset registry con URLs y checksums esperados
DATASET_REGISTRY: Dict[str, Dict] = {
    "bankchurn": {
        "name": "BankChurn Modelling Dataset",
        "url": "https://example.com/datasets/Churn_Modelling.csv",  # Placeholder
        "path": "BankChurn-Predictor/data/Churn_Modelling.csv",
        "checksum": None,  # Se genera automáticamente
        "size_mb": 2,
    },
    "carvision": {
        "name": "CarVision Vehicles Dataset",
        "url": "https://example.com/datasets/vehicles_us.csv",  # Placeholder
        "path": "CarVision-Market-Intelligence/data/vehicles_us.csv",
        "checksum": None,
        "size_mb": 50,
    },
    "telecom": {
        "name": "TelecomAI Users Dataset",
        "url": "https://example.com/datasets/telecom_users.csv",  # Placeholder
        "path": "TelecomAI-Customer-Intelligence/data/telecom_users.csv",
        "checksum": None,
        "size_mb": 5,
    },
    "chicago": {
        "name": "Chicago Taxi Dataset",
        "url": "https://example.com/datasets/taxi_data.csv",  # Placeholder
        "path": "Chicago-Mobility-Analytics/data/taxi_data.csv",
        "checksum": None,
        "size_mb": 30,
    },
    "gold": {
        "name": "Gold Recovery Dataset",
        "url": "https://example.com/datasets/gold_recovery_full.csv",  # Placeholder
        "path": "GoldRecovery-Process-Optimizer/data/gold_recovery_full.csv",
        "checksum": None,
        "size_mb": 20,
    },
    "gaming": {
        "name": "Gaming Market Dataset",
        "url": "https://example.com/datasets/games.csv",  # Placeholder
        "path": "Gaming-Market-Intelligence/data/games.csv",
        "checksum": None,
        "size_mb": 5,
    },
    "oil": {
        "name": "OilWell Location Datasets",
        "url": None,  # Multiple files
        "path": "OilWell-Location-Optimizer/data/",
        "checksum": None,
        "size_mb": 3,
    },
}


def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file from URL with progress indication."""
    try:
        print(f"[*] Descargando: {url}")
        print(f"[*] Destino: {dest_path}")

        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r[*] Progreso: {progress:.1f}%", end="")

        print("\n[✓] Descarga completada")
        return True

    except Exception as e:
        print(f"\n[ERROR] Fallo la descarga: {e}")
        return False


def validate_dataset(project: str) -> bool:
    """Validate dataset existence and integrity."""
    if project not in DATASET_REGISTRY:
        print(f"[ERROR] Proyecto desconocido: {project}")
        return False

    dataset_info = DATASET_REGISTRY[project]
    file_path = PORTFOLIO_ROOT / dataset_info["path"]

    print(f"\n[*] Validando: {dataset_info['name']}")

    # Check existence
    if not file_path.exists():
        print(f"[ERROR] Archivo no encontrado: {file_path}")
        return False

    print(f"[✓] Archivo existe: {file_path}")

    # Check size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"[*] Tamaño: {size_mb:.2f} MB")

    # Compute checksum
    print("[*] Calculando checksum...")
    checksum = compute_checksum(file_path)
    print(f"[*] SHA256: {checksum}")

    # Store or validate checksum
    if dataset_info["checksum"] is None:
        print("[!] Checksum no registrado. Guardando...")
        DATASET_REGISTRY[project]["checksum"] = checksum
    else:
        if checksum != dataset_info["checksum"]:
            print("[ERROR] Checksum no coincide!")
            print(f"  Esperado: {dataset_info['checksum']}")
            print(f"  Obtenido: {checksum}")
            return False
        print("[✓] Checksum válido")

    print("[✓] Dataset validado correctamente")
    return True


def fetch_dataset(project: str) -> bool:
    """Fetch (download) dataset for a project."""
    if project not in DATASET_REGISTRY:
        print(f"[ERROR] Proyecto desconocido: {project}")
        return False

    dataset_info = DATASET_REGISTRY[project]
    file_path = PORTFOLIO_ROOT / dataset_info["path"]

    # Check if already exists
    if file_path.exists():
        print(f"[!] Dataset ya existe: {file_path}")
        print("[*] Usar --validate para verificar integridad")
        return True

    # Download if URL provided
    if dataset_info["url"]:
        return download_file(dataset_info["url"], file_path)
    else:
        print(f"[!] URL no disponible para {project}")
        print(f"[!] Dataset debe estar en: {file_path}")
        return False


def generate_checksums_file() -> None:
    """Generate checksums.json file with all dataset hashes."""
    checksums = {}

    print("\n[*] Generando checksums de todos los datasets...")

    for project, info in DATASET_REGISTRY.items():
        file_path = PORTFOLIO_ROOT / info["path"]

        if file_path.exists() and file_path.is_file():
            checksum = compute_checksum(file_path)
            checksums[project] = {
                "file": str(info["path"]),
                "checksum": checksum,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            }
            print(f"[✓] {project}: {checksum[:16]}...")

    # Save to file
    checksums_file = PORTFOLIO_ROOT / "data" / "checksums.json"
    checksums_file.parent.mkdir(exist_ok=True)

    with open(checksums_file, "w") as f:
        json.dump(checksums, f, indent=2)

    print(f"\n[✓] Checksums guardados en: {checksums_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and validate datasets for ML/MLOps Portfolio")
    parser.add_argument(
        "--project",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        required=True,
        help="Project name or 'all' for all projects",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing datasets instead of downloading",
    )
    parser.add_argument(
        "--generate-checksums",
        action="store_true",
        help="Generate checksums.json file for all datasets",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Portfolio ML/MLOps - Data Fetcher")
    print("=" * 50)

    if args.generate_checksums:
        generate_checksums_file()
        return

    projects = list(DATASET_REGISTRY.keys()) if args.project == "all" else [args.project]

    results = []
    for project in projects:
        if args.validate:
            success = validate_dataset(project)
        else:
            success = fetch_dataset(project)

        results.append((project, success))

    # Summary
    print("\n" + "=" * 50)
    print("Resumen")
    print("=" * 50)

    for project, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {project}")

    # Exit code
    all_success = all(success for _, success in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
