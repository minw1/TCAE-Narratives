#!/bin/bash
#SBATCH --output dsq-hyperjobspos-%A_%3a-%N.out
#SBATCH --array 0-215
#SBATCH --job-name dsq-hyperjobspos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=30:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/frank/wsm32/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/hyperjobspos.txt --status-dir /gpfs/gibbs/project/frank/wsm32/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives

