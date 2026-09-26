#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_selection="${DATASETS:-CA GBA GLA SD}"
read -r -a selected_datasets <<< "${dataset_selection//,/ }"
export DATASETS="${selected_datasets[*]}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    smoke_args=(--datasets "${selected_datasets[@]}")
    if [[ -n "${MODELS:-}" ]]; then
        model_selection="${MODELS//,/ }"
        read -r -a selected_models <<< "${model_selection}"
        smoke_args+=(--models "${selected_models[@]}")
    fi
    if [[ -n "${WINDOWS:-}" ]]; then
        window_selection="${WINDOWS//,/ }"
        read -r -a selected_windows <<< "${window_selection}"
        smoke_args+=(--windows "${selected_windows[@]}")
    fi
    exec "${PYTHON:-python3}" \
        "${SCRIPT_DIR}/smoke_patchstg_four_datasets.py" \
        "${smoke_args[@]}"
fi

exec bash "${SCRIPT_DIR}/run_patchstg_four_datasets.sh"
