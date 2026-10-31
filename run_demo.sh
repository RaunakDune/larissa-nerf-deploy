#!/usr/bin/env bash
# Usage: bash run_demo.sh [GPU_INDEX] [extra args for run_demo.py]
# Creates ./venv on first use, then runs the full demo on the chosen GPU.
set -euo pipefail
GPU="${1:-0}"; shift || true
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  PY=$(command -v python3.11 || command -v python3)
  "$PY" -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  # cu124 wheels still include Pascal (sm_60) kernels, needed for Tesla P40 / Quadro P5000
  ./venv/bin/pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  ./venv/bin/pip install -q -r requirements.txt
fi
CUDA_VISIBLE_DEVICES="$GPU" ./venv/bin/python run_demo.py --device cuda:0 "$@"
