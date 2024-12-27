#!/bin/bash

#SBATCH --job-name=psb_clevrer
#SBATCH --output=logs/psb_clevrer.%j.out
#SBATCH --error=logs/psb_clevrer.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00

echo ${SLURM_NODELIST}

source ~/.bashrc
conda activate baku
export PYTHONPATH=.

srun python train.py \
    data_dir=~/CLEVRER/videos \
    max_video_len=128