#!/usr/bin/env bash
# Run the requested model over forecast horizons; pass extra run.py arguments through.
set -uo pipefail
if (( $# < 4 )); then
  echo "Usage: bash scripts/run_forecasting.sh <tomi|tomi-v> <dataset> <data_path> <window_size> [run.py options]" >&2
  exit 2
fi
model="$1"
dataset="$2"
data_path="$3"
window_size="$4"
shift 4
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
read -r -a horizons <<< "${PRED_LENGTHS:-96 192 336 720}"
log_dir="${LOG_DIR:-logs}"
mkdir -p "$log_dir"
failed=0
for horizon in "${horizons[@]}"; do
  logfile="$log_dir/${model}_${dataset}_H${horizon}_w${window_size}.log"
  echo "Running ${model} ${dataset} H=${horizon} w=${window_size}; log=${logfile}"
  if "$python_bin" -u "$project_dir/run.py" --model "$model" --data "$dataset" \
      --data_path "$data_path" --window_size "$window_size" --pred_len "$horizon" \
      --skip_completed "$@" > "$logfile" 2>&1; then
    tail -n 14 "$logfile"
  else
    failed=$((failed + 1))
    echo "Failed H=${horizon}; continuing to the next horizon. See ${logfile}" >&2
    tail -n 8 "$logfile" >&2
  fi
done
echo "Queue finished: ${#horizons[@]} horizons; ${failed} failures."
if (( failed > 0 )); then exit 1; fi
