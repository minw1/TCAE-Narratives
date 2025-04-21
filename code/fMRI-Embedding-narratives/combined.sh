#!/bin/bash

#SBATCH --job-name=lm_combined
#SBATCH --output=lm_combined_3_29.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a5000:2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

module load CUDA
module load cuDNN
source /gpfs/gibbs/project/frank/wsm32/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/tcae_new/bin/activate

python rt_main.py --num_epochs 50 --predict --perm --data_parallel --checkpoint /home/wsm32/project/wsm_thesis_scratch/narratives/pos_predict_combined_3_29/2-0.1-1e-06-512-large/best_predictor_model.pt --dec_freeze_type thaw_cross --decoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/lm_pretrain_test_3_19/best_lm_pre_model.pt --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/pos_predict_combined_3_29/2-0.1-1e-06-512-large --encoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/0.001-0.5-2-128-0.05-8-16/best_model.pt --report_period 10 --early_stop 100 --lr 0.001 --lr_gamma 0.5 --lr_step_size 15 --n_head 8 --layers 2 --dropout 0.05 --d_latent 128 --n_dec_blocks 2 --dec_dropout 0.1 --weight_decay 0.000001 --d_dec_ff 512 --batch_size 16 --l_vocab 20