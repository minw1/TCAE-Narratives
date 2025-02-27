#!/bin/bash

#SBATCH --job-name=rt_pos_noscans
#SBATCH --output=rt_pos_noscans_2_6.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=40:00:00

module load CUDA
module load cuDNN
source activate tcae
python rt_main.py --num_epochs 100 --predict --data_parallel --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/pos_2_4_lang_only --encoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/0.001-0.5-2-128-0.05-8-16/best_model.pt --report_period 10 --early_stop 100 --lr 0.001 --lr_gamma 0.5 --lr_step_size 15 --n_head 8 --layers 2 --dropout 0.05 --d_latent 128 --batch_size 16 --l_vocab 20 --language_only --dec_dropout 0.2 --n_dec_blocks 2