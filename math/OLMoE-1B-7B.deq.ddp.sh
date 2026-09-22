#!/bin/bash

set -e
cd "$(dirname "$0")"
mkdir -p log checkpoints

# HF cache location: override with HF_CACHE_ROOT if the default disk is small
export HF_HOME=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/hub
export TRANSFORMERS_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/transformers
export HF_DATASETS_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/datasets

export NO_TORCH_COMPILE=1
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

BASE_MODEL='allenai/OLMoE-1B-7B-0924'
DATA_PATH='math_50k.json'

BATCH_SIZE=32
MICRO_BATCH_SIZE=16
NUM_EPOCHS=3

LR=${1:-2e-4}
DEQ_H=${2:-64}
DEQ_NITER=${3:-3}
DEQ_INPUT=${4:-concat}

INPUT_TAG=""
[ "$DEQ_INPUT" != "concat" ] && INPUT_TAG="_${DEQ_INPUT}"
RUN_NAME="OLMoE-1B-7B.math.deq_h${DEQ_H}${INPUT_TAG}_refineonly_n${DEQ_NITER}_lr${LR}"

MASTER_PORT=$((30400 + DEQ_H + 100 * DEQ_NITER))
[ "$DEQ_INPUT" = "eonly" ] && MASTER_PORT=$((MASTER_PORT + 37))
[ "$DEQ_INPUT" = "diff" ] && MASTER_PORT=$((MASTER_PORT + 74))

OUTPUT_DIR="./checkpoints/${RUN_NAME}"
echo "===== training ${RUN_NAME} (h=${DEQ_H}, n=${DEQ_NITER}, input=${DEQ_INPUT}, port=${MASTER_PORT}) ====="

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} finetune.py \
    --base_model "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} --learning_rate ${LR} \
    --cutoff_len 256 --val_set_size 120 --eval_step 80 --save_step 80 \
    --deq_routing_adapter True \
    --deq_hidden_dim ${DEQ_H} --deq_beta 1.0 --deq_n_iter ${DEQ_NITER} --deq_tol 1e-3 \
    --deq_expert_aware_gate False --deq_input_form ${DEQ_INPUT} \
    --wandb_project 'peft-moe' --wandb_run_name "${RUN_NAME}" \
    2>&1 | tee -a "log/${RUN_NAME}.train.log"

echo "===== done ${RUN_NAME} ====="
