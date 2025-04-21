#!/bin/bash

#SBATCH --job-name=rt_test
#SBATCH --output=rt_test_out2.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=16G

source activate tcae
python test_randomtime.py