#!/bin/bash

#SBATCH --job-name=h5
#SBATCH --output=h5.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=week
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G

source activate tcae
python preprocess.py