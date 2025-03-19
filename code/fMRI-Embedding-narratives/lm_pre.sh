#!/bin/bash

#SBATCH --job-name=lmpre
#SBATCH --output=lmpre.txt
#SBATCH --time=12:00:00

conda activate spacy_env
python process_lm_dataset.py