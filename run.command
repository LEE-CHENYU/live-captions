#!/bin/zsh
set -e

cd "${0:A:h}"
exec "${LIVE_CAPTIONS_PYTHON:-/opt/miniconda3/envs/live_captions/bin/python3}" app.py

