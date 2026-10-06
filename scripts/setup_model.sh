#!/bin/bash
# Download and convert the Apache-2.0 SFace model locally.
# This does not install software globally or upload any camera images.
set -euo pipefail
cd "$(dirname "$0")/.."
model_python="${MODEL_PYTHON:-}"
if [ -z "$model_python" ]; then
    for candidate in python3.12 python3.11 python3.10 /usr/bin/python3 python3; do
        candidate_path="$(command -v "$candidate" 2>/dev/null || true)"
        if [ -n "$candidate_path" ] && "$candidate_path" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)' >/dev/null 2>&1; then
            model_python="$candidate_path"
            break
        fi
    done
fi
if [ -z "$model_python" ]; then
    echo 'Python 3.9–3.12 is required for one-time model conversion. Set MODEL_PYTHON to its path.' >&2
    exit 1
fi
model_venv="${MODEL_VENV:-.venv-model}"
if [ ! -x "$model_venv/bin/python" ]; then
    "$model_python" -m venv "$model_venv"
fi
"$model_venv/bin/python" -m pip install --upgrade pip
"$model_venv/bin/python" -m pip install -r requirements-model.txt
"$model_venv/bin/python" tools/fetch_sface.py "$@"
