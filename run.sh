#!/usr/bin/env bash
# Launch Kdenlive Nvidia Renderer (creates the virtualenv on first run).
set -e
# Make file arguments absolute before changing directory.
args=()
for a in "$@"; do
    if [ -e "$a" ]; then args+=("$(readlink -f "$a")"); else args+=("$a"); fi
done
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv/bin/python ]; then
    echo "First run: creating virtual environment and installing dependencies…"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m kdenlive_nvidia_renderer "${args[@]}"
