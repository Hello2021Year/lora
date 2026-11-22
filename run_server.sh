#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage:
  bash run_server.sh check
  bash run_server.sh prepare [--max-length TOKENS]
  bash run_server.sh smoke
  bash run_server.sh train
  bash run_server.sh eval {validation|test} RUN_NAME [CHECKPOINT]

Environment overrides: MODEL, RAW_DATA_DIR, TRAIN_CONDA_PREFIX,
PREPARED_DATA_DIR, CONDA_EXE. Training and evaluation also accept the
environment overrides documented in README.md.
EOF
}

action=${1:-help}
case "$action" in
    help|-h|--help) usage; exit 0 ;;
    check|prepare|smoke|train|eval) shift ;;
    *) usage >&2; exit 2 ;;
esac
if [[ "$action" == check || "$action" == smoke || "$action" == train ]]; then
    if (( $# != 0 )); then
        usage >&2
        exit 2
    fi
fi

export MODEL=${MODEL:-/mnt_upfs/models/qwen3/Qwen3.8-27B}
raw_data_dir=${RAW_DATA_DIR:-/mnt_upfs/models/qwen3/test}
conda_prefix=${TRAIN_CONDA_PREFIX:-/mnt_upfs/conda/swift-qwen38}
prepared_dir=${PREPARED_DATA_DIR:-prepared/compact48k}
archive="$raw_data_dir/jev_datasets_v1.zip"

require_file() {
    if [[ ! -f "$1" || ! -s "$1" || ! -r "$1" ]]; then
        printf 'Required file is missing or unreadable: %s\n' "$1" >&2
        exit 1
    fi
}

runner=(env)
if [[ "${CONDA_PREFIX:-}" != "$conda_prefix" ]]; then
    conda_exe=${CONDA_EXE:-conda}
    if ! command -v "$conda_exe" >/dev/null; then
        printf 'Conda executable was not found. Activate %s or set CONDA_EXE.\n' "$conda_prefix" >&2
        exit 1
    fi
    runner=("$conda_exe" run --no-capture-output --prefix "$conda_prefix")
fi

require_file "$MODEL/config.json"
require_file "$MODEL/tokenizer.json"

case "$action" in
    check)
        require_file "$archive"
        printf 'Model: %s\nData ZIP: %s\nConda prefix: %s\n' "$MODEL" "$archive" "$conda_prefix"
        exec "${runner[@]}" python3 -c '
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

print(f"Python: {sys.executable}", flush=True)
swift_cli = shutil.which("swift")
print(f"Swift CLI: {swift_cli}", flush=True)
missing = []
for package in ("ms-swift", "transformers", "tokenizers", "deepspeed", "qwen-vl-utils"):
    try:
        print(f"{package}: {version(package)}", flush=True)
    except PackageNotFoundError:
        missing.append(package)
if missing:
    sys.exit("Missing packages: " + ", ".join(missing))
if Version(version("ms-swift")) < Version("4.5.0") or Version(version("transformers")) < Version("5.2.0"):
    sys.exit("Qwen3.8 requires ms-swift >= 4.5.0 and transformers >= 5.2.0.")
if not swift_cli:
    sys.exit("Swift CLI was not found in the Conda environment.")

import deepspeed

if sys.argv[1] == "flash_attention_2":
    import flash_attn

gpu_count = torch.cuda.device_count()
print(f"CUDA: {torch.version.cuda}; visible GPUs: {gpu_count}", flush=True)
for index in range(gpu_count):
    print(f"  {index}: {torch.cuda.get_device_name(index)}", flush=True)
if gpu_count < int(sys.argv[2]) or not torch.cuda.is_bf16_supported():
    sys.exit("The configured training GPU count and BF16 support are required.")
' "${ATTN_IMPL:-flash_attention_2}" "${NPROC_PER_NODE:-8}"
        ;;
    prepare)
        require_file "$archive"
        exec "${runner[@]}" python3 prepare_data.py --archive "$archive" \
            --tokenizer "$MODEL/tokenizer.json" --output "$prepared_dir" "$@"
        ;;
    smoke|train)
        require_file "$prepared_dir/audit.json"
        exec "${runner[@]}" bash train.sh "$action" "$prepared_dir"
        ;;
    eval)
        require_file "$prepared_dir/audit.json"
        exec "${runner[@]}" bash evaluate.sh "$prepared_dir" "$@"
        ;;
esac
