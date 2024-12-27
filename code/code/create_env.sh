#!/bin/bash

#SBATCH --job-name=create_my_env
#SBATCH --time=24:00:00
#SBATCH --output=mamba_out.txt
#SBATCH --mem=80G

module load miniconda
mamba env create -f environment-flex.yml --prefix /home/wsm32/.conda/envs/flex_env
