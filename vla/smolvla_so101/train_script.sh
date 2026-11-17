#!/usr/bin/env bash
set -euo pipefail

# Defaults: override from CLI with, for example:
#   STEPS=20000 BATCH_SIZE=4 DEVICE=auto ./train_script.sh
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
DEVICE="${DEVICE:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$DEVICE" == "auto" ]]; then
  if python -c 'import sys; sys.exit(0 if __import__("torch").cuda.is_available() else 1)' >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

# Use local ModelScope-downloaded files; do not hit Hugging Face.
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_LOCAL_FILES_ONLY=1
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_MODULES_CACHE=/tmp/hf_modules_cache
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

# This directory links to the local base checkpoint and has local configs
# pointing to the local SmolVLM weights/tokenizer.
POLICY_PATH="$SCRIPT_DIR/local_smolvla_base"
DATASET_ROOT="/home/user/ubuntu/datasets/svla_so101_pickplace"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/smolvla_so101_local}"

lerobot-train \
  --policy.path="$POLICY_PATH" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/svla_so101_pickplace" \
  --dataset.root="$DATASET_ROOT" \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="smolvla_so101_local_training" \
  --wandb.enable=false
