#!/bin/bash

#SBATCH --job-name=rt_tcae_large
#SBATCH --output=rt_tcae_large_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=40:00:00

module load CUDA
module load cuDNN
source activate tcae
python rt_main.py --num_epochs 150 --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/1-9 --data_parallel --report_period 10 --early_stop 100 --lr 0.05 --lr_gamma 0.5 --lr_step_size 15 --n_head 8 --layers 4 --dropout 0.1 --d_latent 128
