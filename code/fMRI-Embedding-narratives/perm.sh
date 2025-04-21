#!/bin/bash

#SBATCH --job-name=permtest
#SBATCH --output=permtest.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

module load CUDA
module load cuDNN
source /gpfs/gibbs/project/frank/wsm32/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/tcae_new/bin/activate

python perm_tester.py