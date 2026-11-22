#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mode=${1:-}
if [[ $# -ne 2 || ( "$mode" != smoke && "$mode" != train ) ]]; then
    echo "Usage: bash train.sh {smoke|train} PREPARED_DATA_DIR" >&2
    exit 2
fi
data_dir=$2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if (( NNODES > 1 )); then
    : "${MASTER_ADDR:?Set MASTER_ADDR to the training-network IP of node 0}"
    : "${NODE_RANK:?Set NODE_RANK to 0 on the first node and 1 on the second}"
fi

max_length=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_length"])' "$data_dir/audit.json")
model=${MODEL:-Qwen/Qwen3.8-27B}
attn_impl=${ATTN_IMPL:-flash_attention_2}
model_source=(--use_hf true)
if [[ "$model" == Qwen/Qwen3.8-27B ]]; then
    model_source+=(--model_revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0)
fi
python3 - "$NPROC_PER_NODE" "$attn_impl" <<'PY'
import sys
from importlib.metadata import version

import torch
from packaging.version import Version

if Version(version("ms-swift")) < Version("4.5.0"):
    sys.exit("Install ms-swift >= 4.5.0 for Qwen3.8-27B.")
if torch.cuda.device_count() < int(sys.argv[1]) or not torch.cuda.is_bf16_supported():
    sys.exit("This recipe requires the configured number of CUDA GPUs with BF16 support.")
import deepspeed
if sys.argv[2] == "flash_attention_2":
    import flash_attn
PY

output_dir=${OUTPUT_DIR:-outputs/$(basename "$data_dir")-$mode}
if [[ "$mode" == smoke ]]; then
    smoke_steps=${SMOKE_STEPS:-3}
    mode_args=(
        --dataset "$data_dir/smoke.jsonl" --max_steps "$smoke_steps"
        --gradient_accumulation_steps 1 --eval_strategy no
        --save_strategy steps --save_steps "$smoke_steps"
    )
else
    world_size=$((NPROC_PER_NODE * NNODES))
    gradient_accumulation=$(((16 + world_size - 1) / world_size))
    mode_args=(
        --dataset "$data_dir/train.jsonl" --val_dataset "$data_dir/validation.jsonl"
        --num_train_epochs "${EPOCHS:-1}"
        --gradient_accumulation_steps "${GRAD_ACCUM:-$gradient_accumulation}"
        --eval_strategy epoch --save_strategy epoch
    )
fi

exec swift sft \
    --model "$model" "${model_source[@]}" --model_type qwen3_5 --template qwen3_8 \
    --tuner_type lora --torch_dtype bfloat16 --deepspeed zero3 \
    --target_modules all-linear --freeze_vit true --freeze_aligner true \
    --lora_rank "${LORA_RANK:-8}" --lora_alpha "${LORA_ALPHA:-16}" \
    --learning_rate "${LEARNING_RATE:-5e-5}" --warmup_ratio 0.05 \
    --loss_scale last_round+ignore_empty_think --use_logits_to_keep true \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --max_length "$max_length" --truncation_strategy delete \
    --gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --attn_impl "$attn_impl" --packing false --lazy_tokenize false --strict true \
    --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
    --dataloader_num_workers 2 --seed 42 --data_seed 42 \
    --logging_steps 1 --report_to none --save_total_limit 2 \
    --create_checkpoint_symlink true --output_dir "$output_dir" "${mode_args[@]}"
