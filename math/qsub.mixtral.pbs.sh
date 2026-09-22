#!/bin/bash
#PBS -N mixtral_deq
#PBS -l select=1:ncpus=8:ngpus=2:gpu_model=H200:mem=200gb
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -o ./log/

#
#
#
#   qsub -v DEQ_H=8  qsub.mixtral.pbs.sh
#   qsub -v DEQ_H=64,DEQ_XLAYER=off qsub.mixtral.pbs.sh

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
conda activate ml_env

cd $(pwd)/math

LR=${LR:-2e-4}
DEQ_H=${DEQ_H:-8}
DEQ_NITER=${DEQ_NITER:-3}
DEQ_INPUT=${DEQ_INPUT:-concat}
DEQ_XLAYER=${DEQ_XLAYER:-on}

nvidia-smi --query-gpu=name,memory.total --format=csv

bash Mixtral-8x7B.deq.sh "${LR}" "${DEQ_H}" "${DEQ_NITER}" "${DEQ_INPUT}" "${DEQ_XLAYER}"
