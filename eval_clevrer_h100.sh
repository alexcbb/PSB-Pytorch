#!/bin/bash

#SBATCH --job-name=psb_clevrer
#SBATCH --output=logs/psb_clevrer.%j.out
#SBATCH --error=logs/psb_clevrer.%j.err
#SBATCH -A uli@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 60:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

source ~/.bashrc

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=.

srun python eval.py \
    data_dir=~/CLEVRER/videos \
    max_video_len=128 \
    +checkpoint_path=#TOCHANGE