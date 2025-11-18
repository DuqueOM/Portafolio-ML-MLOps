#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-configs/default.yaml}

python data/preprocess.py --config "$CONFIG"
python main.py --mode train --config "$CONFIG" --seed 42
python evaluate.py --config "$CONFIG"
