#!/bin/bash

#SBATCH --job-name=rt_tcae
#SBATCH --output=rt_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

module load CUDA
module load cuDNN
source activate tcae
python rt_main.py --num_epochs 100 --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_models --data_parallel --report_period 10 --early_stop 20 --lr 0.01 --layers 2 --dropout 0.1
