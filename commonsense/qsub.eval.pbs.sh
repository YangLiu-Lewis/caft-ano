#!/bin/bash
#PBS -N deq_eval
#PBS -l select=1:ncpus=4:ngpus=2:gpu_model=H200:mem=160gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o ./log/

#   qsub -v CKPT=OLMoE-1B-7B.deq_h128_refineonly_n3_lr2e-4 qsub.eval.pbs.sh

__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
case "${CKPT:-}" in
    *Qwen3*) conda activate qwen3moe ;;
    *)       conda activate ml_env   ;;
esac

cd $(pwd)/commonsense

: "${CKPT:?pass -v CKPT=<checkpoint> to select the checkpoint to evaluate}"

bash eval_deq.sh "${CKPT}" 0 1
