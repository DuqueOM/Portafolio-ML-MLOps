#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:-example_payload.json}
python main.py --mode predict --config configs/config.yaml --input_json "$INPUT"
