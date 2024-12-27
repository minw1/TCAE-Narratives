#!/bin/bash

#SBATCH --job-name=narr_tcae
#SBATCH --output=narr_out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

module load CUDA
module load cuDNN
source activate tcae
python narr_main.py --num_epochs 100 --exp_dir auto --data_parallel --report_period 10 --early_stop 20 --lr 0.01 --checkpoint /home/wsm32/palmer_scratch/wsm_thesis_scratch/narratives/models/model.pt
