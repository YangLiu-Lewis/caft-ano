#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:00
#SBATCH --job-name=OLMoE-1B-7B.ddp.train
#SBATCH --output=log/OLMoE-1B-7B.ddp.%j.train.slurm.log
#SBATCH --mail-type=ALL


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
DATA_PATH='commonsense_170k.json'

BATCH_SIZE=32
MICRO_BATCH_SIZE=16
NUM_EPOCHS=3

METHOD=${1:-deq}
LR_ARG=${2:-}
DEQ_H=${3:-64}
DEQ_NITER=${4:-3}
DEQ_INPUT=${5:-concat}
DEQ_XLAYER=${6:-off}

if [ "$METHOD" = "perft" ]; then
    LR=${LR_ARG:-1e-5}
    RUN_NAME="OLMoE-1B-7B.perft_pa16_lr${LR}"
    METHOD_ARGS="--shared_routing_adapter True \
        --shared_routing_adapter_num_experts 8 \
        --shared_routing_adapter_num_experts_per_tok 1 \
        --adapter_type Parallel_Adapter --hidden_dim 16"
    MASTER_PORT=29501
elif [ "$METHOD" = "deq" ]; then
    LR=${LR_ARG:-2e-4}
    INPUT_TAG=""
    [ "$DEQ_INPUT" != "concat" ] && INPUT_TAG="_${DEQ_INPUT}"
    [ "$DEQ_XLAYER" = "on" ] && XLAYER_TAG="_xlayer" || XLAYER_TAG=""
    [ "$DEQ_XLAYER" = "on" ] && XLAYER_VAL="True" || XLAYER_VAL="False"
    RUN_NAME="OLMoE-1B-7B.deq_h${DEQ_H}${INPUT_TAG}${XLAYER_TAG}_refineonly_n${DEQ_NITER}_lr${LR}"
    METHOD_ARGS="--deq_routing_adapter True \
        --deq_hidden_dim ${DEQ_H} --deq_beta 1.0 --deq_n_iter ${DEQ_NITER} --deq_tol 1e-3 \
        --deq_expert_aware_gate False --deq_input_form ${DEQ_INPUT} \
        --deq_cross_layer ${XLAYER_VAL}"
    MASTER_PORT=$((29400 + DEQ_H + 100 * DEQ_NITER))
    [ "$DEQ_INPUT" = "eonly" ] && MASTER_PORT=$((MASTER_PORT + 37))
    [ "$DEQ_INPUT" = "diff" ] && MASTER_PORT=$((MASTER_PORT + 74))
else
    echo "unknown METHOD: $METHOD (use 'perft' or 'deq')"; exit 1
fi

OUTPUT_DIR="./checkpoints/${RUN_NAME}"
echo "===== training ${RUN_NAME} (method=${METHOD}, h=${DEQ_H}, port=${MASTER_PORT}) ====="

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} finetune.py \
    --base_model "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} --learning_rate ${LR} \
    --cutoff_len 256 --val_set_size 120 --eval_step 80 --save_step 80 \
    ${METHOD_ARGS} \
    --wandb_project 'peft-moe' --wandb_run_name "${RUN_NAME}" \
    2>&1 | tee -a "log/${RUN_NAME}.train.log"

echo "===== done ${RUN_NAME} ====="
