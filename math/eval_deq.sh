#!/bin/bash

set -e
cd "$(dirname "$0")"
mkdir -p log

export HF_HOME=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/hub
export TRANSFORMERS_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/transformers
export HF_DATASETS_CACHE=${HF_CACHE_ROOT:-$HOME/.cache/huggingface}/datasets

export NO_TORCH_COMPILE=1

CKPT_NAME=${1:?usage: bash eval_deq.sh <checkpoint> [gpu0] [gpu1] [batch_size]}
GPU0=${2:-0}
GPU1=${3:-1}
PEFT_MODEL="checkpoints/${CKPT_NAME}"
TAG=${CKPT_NAME//\//_}

case "$CKPT_NAME" in
    *Mixtral*) BASE_MODEL='mistralai/Mixtral-8x7B-v0.1'; DEFAULT_BS=24 ;;
    *Qwen3*)   BASE_MODEL='Qwen/Qwen3-30B-A3B-Base';   DEFAULT_BS=16 ;;
    *)         BASE_MODEL='allenai/OLMoE-1B-7B-0924';   DEFAULT_BS=8  ;;
esac
BATCH_SIZE=${4:-$DEFAULT_BS}
echo "===== base model: ${BASE_MODEL}  batch_size: ${BATCH_SIZE} ====="

if [ ! -f "${PEFT_MODEL}/model.safetensors" ]; then
    echo "${PEFT_MODEL}/model.safetensors not found -- check the checkpoint name"; exit 1
fi

GROUP0=(gsm8k AddSub SingleEq)
GROUP1=(AQuA MultiArith SVAMP)

set -o pipefail 2>/dev/null || true
rm -f "log/${TAG}.eval.FAILED"

run_group() {
    local gpu_id=$1; shift
    local datasets=("$@")
    for ds in "${datasets[@]}"; do
        echo "[GPU ${gpu_id}] ===== evaluating ${CKPT_NAME} on ${ds} ====="
        CUDA_VISIBLE_DEVICES=${gpu_id} python math_evaluate.py \
            --dataset ${ds} \
            --base_model ${BASE_MODEL} \
            --peft_model ${PEFT_MODEL} \
            --name ${TAG} \
            --batch_size ${BATCH_SIZE} --max_new_tokens 256 \
            2>&1 | tee -a "log/${TAG}.eval.${ds}.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" != "0" ]; then
            echo "[GPU ${gpu_id}] !!!!! ${ds} failed (exit code ${rc}) -- result incomplete, do not use !!!!!"
            echo "$ds" >> "log/${TAG}.eval.FAILED"
        fi
    done
}

run_group ${GPU0} "${GROUP0[@]}" &
PID0=$!
run_group ${GPU1} "${GROUP1[@]}" &
PID1=$!

wait ${PID0} ${PID1}

if [ -s "log/${TAG}.eval.FAILED" ]; then
    echo "!!!!! the following tasks failed; rerun with a smaller batch_size (4th argument):"
    sort -u "log/${TAG}.eval.FAILED" | sed 's/^/      /'
    echo "!!!!! e.g. bash eval_deq.sh ${CKPT_NAME} 0 1 8"
fi
echo "===== all 6 tasks evaluated ====="
