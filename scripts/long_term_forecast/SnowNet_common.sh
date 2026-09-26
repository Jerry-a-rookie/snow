#!/usr/bin/env bash
set -euo pipefail

run_snownet() {
  local name="$1"
  local data="$2"
  local root_path="$3"
  local data_path="$4"
  local freq="$5"
  local enc_in="$6"
  local e_layers="$7"
  local d_model="$8"
  local d_core="$9"
  local d_ff="${10}"
  local learning_rate="${11}"
  local train_epochs="${12}"
  local patience="${13}"
  local batch_size="${14}"
  local snow_clusters="${15}"
  local pred_len="${16}"
  local result_path="${17}"
  local use_norm="${18:-}"
  local dry_run="${19:-0}"

  local common_args=(
    --task_name long_term_forecast
    --is_training 1
    --root_path "$root_path"
    --data_path "$data_path"
    --model SnowNet
    --data "$data"
    --result_data "$name"
    --features M
    --target OT
    --freq "$freq"
    --seq_len 96
    --label_len 48
    --pred_len "$pred_len"
    --e_layers "$e_layers"
    --enc_in "$enc_in"
    --dec_in "$enc_in"
    --c_out "$enc_in"
    --d_model "$d_model"
    --d_core "$d_core"
    --d_ff "$d_ff"
    --learning_rate "$learning_rate"
    --lradj cosine
    --train_epochs "$train_epochs"
    --patience "$patience"
    --des Exp
    --itr 1
    --result_path "$result_path"
    --batch_size "$batch_size"
  )

  if [[ -n "$use_norm" ]]; then
    common_args+=(--use_norm "$use_norm")
  fi

  local snow_id="${name}_96_${pred_len}_SnowNet"
  local snow_cmd=(
    python -u run.py
    "${common_args[@]}"
    --model_id "$snow_id"
    --snow_clusters "$snow_clusters"
  )
  echo "${snow_cmd[*]}"
  if [[ "$dry_run" != "1" ]]; then
    "${snow_cmd[@]}"
  fi
}

print_results() {
  local result_path="$1"
  if [[ -f "$result_path" ]]; then
    python - "$result_path" <<'PY'
import sys
import warnings

warnings.filterwarnings(
    'ignore',
    message='Pandas requires version .*',
    category=UserWarning,
)

import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
cols = ['data', 'model_id', 'model', 'pred_len', 'mse', 'mae']
print(df[cols].sort_values(['data', 'pred_len', 'mse']).to_string(index=False))
PY
  fi
}

write_snapshot() {
  local result_path="$1"
  local snapshot_path="$2"
  local data_name="$3"
  local pred_len="$4"

  if [[ ! -f "$result_path" ]]; then
    return
  fi

  mkdir -p "$(dirname "$snapshot_path")"
  python - "$result_path" "$snapshot_path" "$data_name" "$pred_len" <<'PY'
import sys
import warnings

warnings.filterwarnings(
    'ignore',
    message='Pandas requires version .*',
    category=UserWarning,
)

import pandas as pd

result_path, snapshot_path, data_name, pred_len = sys.argv[1:]
df = pd.read_csv(result_path)
df = df[(df['data'].astype(str) == data_name) & (df['pred_len'].astype(str) == str(pred_len))]
if df.empty:
    raise SystemExit(0)
cols = ['data', 'model_id', 'model', 'pred_len', 'mse', 'mae']
df[cols].sort_values('mse').to_csv(snapshot_path, index=False)
print(df[cols].sort_values('mse').to_string(index=False))
print(f"saved: {snapshot_path}")
PY
}
