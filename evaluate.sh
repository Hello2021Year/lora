#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
split=${2:-}
run_name=${3:-}
if [[ $# -lt 3 || $# -gt 4 || ( "$split" != validation && "$split" != test ) || ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Usage: bash evaluate.sh DATA_DIR {validation|test} RUN_NAME [CHECKPOINT]" >&2
    exit 2
fi
data_dir=$1
holdout="$data_dir/$split.jsonl"
train_file=${TRAIN_FILE:-$data_dir/train.jsonl}
max_length=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_length"])' "$data_dir/audit.json")
model=${MODEL:-Qwen/Qwen3.8-27B}
model_source=(--use_hf true)
if [[ "$model" == Qwen/Qwen3.8-27B ]]; then
    model_source+=(--model_revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0)
fi
if [[ $# -eq 4 ]]; then
    test -f "$4/adapter_config.json"
    model_source+=(--adapters "$4")
fi
test -s "$holdout"
test -s "$train_file"
mkdir -p results
run_dir="results/$run_name"
mkdir "$run_dir"

# A single process can use multiple visible GPUs for model placement.
unset NPROC_PER_NODE NNODES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
swift infer \
    --model "$model" "${model_source[@]}" \
    --model_type qwen3_5 --template qwen3_8 --load_args false --load_data_args false \
    --infer_backend transformers --torch_dtype bfloat16 --device_map auto \
    --attn_impl "${ATTN_IMPL:-flash_attention_2}" --max_batch_size 1 \
    --val_dataset "$holdout" --val_dataset_shuffle false --stream false \
    --enable_thinking false --temperature 0 --max_new_tokens 8 \
    --max_length "$max_length" --result_path "$run_dir/predictions.jsonl"

python3 score_eval.py "$run_dir/predictions.jsonl" \
    --expected "$holdout" --train "$train_file" | tee "$run_dir/metrics.json"
