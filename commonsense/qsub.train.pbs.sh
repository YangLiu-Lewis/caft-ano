#!/bin/bash
#PBS -N deq_train
#PBS -l select=1:ncpus=4:ngpus=2:gpu_model=H200:mem=160gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o ./log/

#   qsub -v DEQ_H=16 qsub.train.pbs.sh                       # h-sweep: h16
#   qsub -v DEQ_H=32,DEQ_NITER=3 qsub.train.pbs.sh          # h-sweep: h32
#   qsub -v METHOD=perft,LR=1e-5 qsub.train.pbs.sh          # perft baseline

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

cd $(pwd)/commonsense

METHOD=${METHOD:-deq}
LR=${LR:-2e-4}
DEQ_H=${DEQ_H:-64}
DEQ_NITER=${DEQ_NITER:-3}
DEQ_INPUT=${DEQ_INPUT:-concat}
DEQ_XLAYER=${DEQ_XLAYER:-off}

bash OLMoE-1B-7B.ddp.sh "${METHOD}" "${LR}" "${DEQ_H}" "${DEQ_NITER}" "${DEQ_INPUT}" "${DEQ_XLAYER}"
