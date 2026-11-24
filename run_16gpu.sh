#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

rank=${1:-}
mode=${2:-smoke}
size=${3:-4k}
if [[ $# -gt 3 || ! "$rank" =~ ^[01]$ || ! "$mode" =~ ^(check|smoke|train)$ || ! "$size" =~ ^(4k|8k|64k)$ ]]; then
    echo 'Usage: bash run_16gpu.sh <0|1> [check|smoke|train] [4k|8k|64k]' >&2
    echo 'Set MASTER_ADDR to the training-network IP of node 0 on both nodes.' >&2
    exit 2
fi

export NODE_RANK="$rank" NNODES=2 NPROC_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29561}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}" SMOKE_STEPS=3 GRAD_ACCUM=1 EPOCHS=1
case "$size" in
    64k) export DATA_VERSION=v2 EXPECTED_MAX_LENGTH=65536; name=dataV2-compact64k ;;
    8k) export DATA_VERSION=v1 EXPECTED_MAX_LENGTH=8192; name=compact8k ;;
    4k) export DATA_VERSION=v1 EXPECTED_MAX_LENGTH=4096; name=compact4k ;;
esac
export PREPARED_DATA_DIR="prepared/$name"
export OUTPUT_DIR="outputs/$name-16gpu-$mode"

exec bash run_server.sh "$mode"
