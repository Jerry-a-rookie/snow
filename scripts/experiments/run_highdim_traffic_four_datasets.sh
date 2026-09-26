#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
TRAFFIC_ROOT="${TRAFFIC_ROOT:-${PROJECT_ROOT}/dataset/highdim_traffic}"
# TRAFFIC_ROOT is the converted-data root.  When the raw traffic NPZ files
# are stored there as flowca.npz/flowgba.npz/flowgla.npz/flowsd.npz, the
# missing converted dataset directories are prepared automatically below.
TRAFFIC_SOURCE_ROOT="${TRAFFIC_SOURCE_ROOT:-${TRAFFIC_ROOT}}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results/highdim_traffic_four_datasets}"
SEED="${SEED:-2021}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-32}"
D_CORE="${D_CORE:-32}"
D_FF="${D_FF:-64}"
E_LAYERS="${E_LAYERS:-1}"
N_HEADS="${N_HEADS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DROPOUT="${DROPOUT:-0.0}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GPU="${GPU:-0}"

ALL_MODELS=(
    NLinear RLinear GRU GRU_CI GRU_CD TCN CycleNet PatchTST TSMixer XLinear Amplifier
    SegRNN iTransformer NLinear_Snow RLinear_Snow GRU_Snow GRU_CI_Snow GRU_CD_Snow TCN_Snow
    CycleNet_Snow PatchTST_Snow TSMixer_Snow XLinear_Snow Amplifier_Snow
    SegRNN_Snow iTransformer_Snow iTransformer_Snow_Attn1
    iTransformer_Snow_AllAttn
)
ALL_DATASETS=(CA GBA GLA SD)
ALL_WINDOWS=(6 12)

model_selection="${MODELS:-${ALL_MODELS[*]}}"
dataset_selection="${DATASETS:-${ALL_DATASETS[*]}}"
window_selection="${WINDOWS:-${ALL_WINDOWS[*]}}"
read -r -a SELECTED_MODELS <<< "${model_selection//,/ }"
read -r -a SELECTED_DATASETS <<< "${dataset_selection//,/ }"
read -r -a SELECTED_WINDOWS <<< "${window_selection//,/ }"

contains() {
    local wanted="$1"
    shift
    local value
    for value in "$@"; do
        [[ "${value}" == "${wanted}" ]] && return 0
    done
    return 1
}

for model in "${SELECTED_MODELS[@]}"; do
    contains "${model}" "${ALL_MODELS[@]}" || {
        echo "Unknown model: ${model}" >&2
        exit 1
    }
done
for dataset in "${SELECTED_DATASETS[@]}"; do
    contains "${dataset}" "${ALL_DATASETS[@]}" || {
        echo "Unknown dataset: ${dataset}" >&2
        exit 1
    }
done
for window in "${SELECTED_WINDOWS[@]}"; do
    contains "${window}" "${ALL_WINDOWS[@]}" || {
        echo "Window must be 6 or 12, received: ${window}" >&2
        exit 1
    }
done
if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
    echo "SHARD_INDEX must be in [0, SHARD_COUNT)." >&2
    exit 1
fi

dataset_config() {
    case "$1" in
        CA) CHANNELS=8600; DEFAULT_BATCH=1 ;;
        GBA) CHANNELS=2352; DEFAULT_BATCH=2 ;;
        GLA) CHANNELS=3834; DEFAULT_BATCH=1 ;;
        SD) CHANNELS=716; DEFAULT_BATCH=8 ;;
        *) return 1 ;;
    esac
}

if [[ "${DRY_RUN}" != "1" ]]; then
    for dataset in "${SELECTED_DATASETS[@]}"; do
        converted_ready=1
        for file in flow.npy meta.csv manifest.json; do
            [[ -f "${TRAFFIC_ROOT}/${dataset}/${file}" ]] || converted_ready=0
        done

        if (( converted_ready == 0 )); then
            case "${dataset}" in
                CA) raw_name="flowca.npz"; meta_name="CA/ca_meta.csv" ;;
                GBA) raw_name="flowgba.npz"; meta_name="GBA/gba_meta.csv" ;;
                GLA) raw_name="flowgla.npz"; meta_name="GLA/gla_meta.csv" ;;
                SD) raw_name="flowsd.npz"; meta_name="SD/sd_meta.csv" ;;
            esac
            raw_path="${TRAFFIC_SOURCE_ROOT}/${raw_name}"
            meta_path="${TRAFFIC_SOURCE_ROOT}/${meta_name}"
            if [[ -f "${raw_path}" && -f "${meta_path}" ]]; then
                prepare_script="${PROJECT_ROOT}/scripts/data/prepare_highdim_traffic.py"
                [[ -f "${prepare_script}" ]] || {
                    echo "High-dimensional traffic converter not found: ${prepare_script}" >&2
                    exit 1
                }
                echo "Preparing ${dataset}: ${raw_path} -> ${TRAFFIC_ROOT}/${dataset}"
                "${PYTHON_BIN}" "${prepare_script}" \
                    --source-root "${TRAFFIC_SOURCE_ROOT}" \
                    --output-root "${TRAFFIC_ROOT}" \
                    --datasets "${dataset}" \
                    --overwrite
            else
                echo "Missing high-dimensional traffic data for ${dataset}. Expected either converted files under ${TRAFFIC_ROOT}/${dataset}/ or raw files ${raw_path} and ${meta_path}." >&2
                exit 1
            fi
        fi

        for file in flow.npy meta.csv manifest.json; do
            [[ -f "${TRAFFIC_ROOT}/${dataset}/${file}" ]] || {
                echo "Missing converted dataset file after preparation: ${TRAFFIC_ROOT}/${dataset}/${file}" >&2
                exit 1
            }
        done
    done
fi

if (( SHARD_COUNT > 1 )); then
    printf -v shard_name 'shard_%02d_of_%02d' "${SHARD_INDEX}" "${SHARD_COUNT}"
    RESULT_DIR="${RESULT_DIR}/${shard_name}"
fi
mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/commands" "${RESULT_DIR}/checkpoints"
RESULT_PATH="${RESULT_DIR}/metrics.csv"
MANIFEST_PATH="${RESULT_DIR}/manifest.csv"
printf 'JobIndex,Dataset,Model,Seed,SeqLen,PredLen,BatchSize,Gpu,ModelId,Status,LogPath,Command\n' > "${MANIFEST_PATH}"

declare -A COMPLETED=()
if [[ "${RESUME}" == "1" && -f "${RESULT_PATH}" ]]; then
    while IFS=$'\t' read -r data model seed seq_len pred_len; do
        [[ -n "${data}" ]] || continue
        COMPLETED["${data}|${model}|${seed}|${seq_len}|${pred_len}"]=1
    done < <(
        "${PYTHON_BIN}" -c '
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        print(row["data"], row["model"], row["seed"], row["seq_len"], row["pred_len"], sep="\t")
' "${RESULT_PATH}"
    )
fi

shell_join() {
    printf '%q ' "$@"
}

full_count=$((
    ${#SELECTED_DATASETS[@]} * ${#SELECTED_WINDOWS[@]} * ${#SELECTED_MODELS[@]}
))
selected_count=0
for ((i=0; i<full_count; i++)); do
    (( i % SHARD_COUNT == SHARD_INDEX )) && selected_count=$((selected_count + 1))
done
echo "High-dimensional traffic jobs: ${selected_count} selected / ${full_count} total"
echo "Models: ${#SELECTED_MODELS[@]}; datasets: ${SELECTED_DATASETS[*]}; windows: ${SELECTED_WINDOWS[*]}"
echo "Shard: ${SHARD_INDEX}/${SHARD_COUNT}; GPU: ${GPU}; manifest: ${MANIFEST_PATH}"

job_index=0
run_number=0
for dataset in "${SELECTED_DATASETS[@]}"; do
    dataset_config "${dataset}"
    for window in "${SELECTED_WINDOWS[@]}"; do
        for model in "${SELECTED_MODELS[@]}"; do
            current_index="${job_index}"
            job_index=$((job_index + 1))
            (( current_index % SHARD_COUNT == SHARD_INDEX )) || continue
            run_number=$((run_number + 1))

            batch_size="${BATCH_SIZE_OVERRIDE}"
            (( batch_size > 0 )) || batch_size="${DEFAULT_BATCH}"
            model_id="${dataset}_${model}_${window}to${window}_seed${SEED}"
            log_path="${RESULT_DIR}/logs/${model_id}.log"
            command_path="${RESULT_DIR}/commands/${model_id}.sh"
            key="${dataset}|${model}|${SEED}|${window}|${window}"
            status="pending"
            [[ "${RESUME}" == "1" && -n "${COMPLETED[${key}]:-}" ]] && status="completed"

            args=(
                -u run.py
                --task_name long_term_forecast
                --is_training 1
                --model_id "${model_id}"
                --model "${model}"
                --seed "${SEED}"
                --data HighDimTraffic
                --result_data "${dataset}"
                --root_path "${TRAFFIC_ROOT}/${dataset}"
                --data_path flow.npy
                --features M
                --freq 15min
                --seq_len "${window}"
                --label_len 0
                --pred_len "${window}"
                --enc_in "${CHANNELS}"
                --dec_in "${CHANNELS}"
                --c_out "${CHANNELS}"
                --d_model "${D_MODEL}"
                --d_core "${D_CORE}"
                --d_ff "${D_FF}"
                --e_layers "${E_LAYERS}"
                --n_heads "${N_HEADS}"
                --hidden_size "${HIDDEN_SIZE}"
                --hidden_dim "${HIDDEN_DIM}"
                --individual 0
                --dropout "${DROPOUT}"
                --patch_len 3
                --stride 3
                --seg_len 3
                --cycle 96
                --use_norm 1
                --train_epochs "${TRAIN_EPOCHS}"
                --patience "${PATIENCE}"
                --batch_size "${batch_size}"
                --num_workers "${NUM_WORKERS}"
                --learning_rate "${LEARNING_RATE}"
                --lradj type1
                --itr 1
                --gpu "${GPU}"
                --use_amp
                --des HighDimTrafficFourDatasets
                --result_path "${RESULT_PATH}"
                --checkpoints "${RESULT_DIR}/checkpoints"
            )
            command_text="$(shell_join "${PYTHON_BIN}" "${args[@]}")"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
                "${current_index}" "${dataset}" "${model}" "${SEED}" "${window}" \
                "${window}" "${batch_size}" "${GPU}" "${model_id}" "${status}" \
                "${log_path}" "${command_text}" >> "${MANIFEST_PATH}"

            printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nexec %q ' \
                "${PROJECT_ROOT}" "${PYTHON_BIN}" > "${command_path}"
            printf '%q ' "${args[@]}" >> "${command_path}"
            printf '\n' >> "${command_path}"
            chmod +x "${command_path}"

            echo "[${run_number}/${selected_count}] ${dataset} | ${model} | ${window}->${window} | ${status}"
            [[ "${status}" == "completed" || "${DRY_RUN}" == "1" ]] && continue

            set +e
            "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${log_path}"
            exit_code=${PIPESTATUS[0]}
            set -e
            (( exit_code == 0 )) || exit "${exit_code}"
        done
    done
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry-run complete. No training was started."
else
    echo "Selected high-dimensional traffic experiments complete."
fi
