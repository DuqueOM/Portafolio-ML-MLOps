#!/usr/bin/env bash
set -euo pipefail
python main.py --mode train --config configs/default.yaml "$@"
