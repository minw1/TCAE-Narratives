batch_size = 16
lr = 0.001
lr_decay = 0.5

small_encoder = {
    "lr" : 0.001,
    "lr_decay" : 0.5,
    "layer_count" : 1,
    "latent" : 64,
    "head_count" : 4,
    "dropout" : 0.05,
    "batch_size" : 16,
    "kind" : "small",
}

large_encoder = {
    "lr" : 0.001,
    "lr_decay" : 0.5,
    "layer_count" : 2,
    "latent" : 128,
    "head_count" : 8,
    "dropout" : 0.05,
    "batch_size" : 16,
    "kind" : "large",
}

encoders = [small_encoder, large_encoder]
'''

with open("hyperjobspos.txt", "w") as file:
    for dec_blocks in [1, 2]:
        for dec_dropout in [0.1, 0.3, 0.4]:
            for wd in [1e-6, 1e-4, 1e-3]:
                for dec_ff in [128, 256, 512]:
                    for encoder in encoders:
                        #The scans model
                         #The noscans model
                        file.write("module load CUDA; ")
                        file.write("module load cuDNN; ")
                        file.write("source activate tcae; ")
                        file.write(f"python rt_main.py --num_epochs 100 --predict --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/pos_predict/{dec_blocks}-{dec_dropout}-{wd}-{dec_ff}-{encoder["kind"]} --encoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/{encoder["lr"]}-{encoder["lr_decay"]}-{encoder["layer_count"]}-{encoder["latent"]}-{encoder["dropout"]}-{encoder["head_count"]}-{encoder["batch_size"]}/best_model.pt --data_parallel --report_period 10 --early_stop 100 --lr {lr} --lr_gamma {lr_decay} --lr_step_size 15 --n_head {encoder["head_count"]} --layers {encoder["layer_count"]} --dropout {encoder["dropout"]} --d_latent {encoder["latent"]} --n_dec_blocks {dec_blocks} --dec_dropout {dec_dropout} --weight_decay {wd} --d_dec_ff {dec_ff} --batch_size {batch_size} --l_vocab 20\n")

                        #The noscans model
                        file.write("module load CUDA; ")
                        file.write("module load cuDNN; ")
                        file.write("source activate tcae; ")
                        file.write(f"python rt_main.py --num_epochs 100 --predict --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/pos_predict/{dec_blocks}-{dec_dropout}-{wd}-{dec_ff}-{encoder["kind"]}-lo --encoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/{encoder["lr"]}-{encoder["lr_decay"]}-{encoder["layer_count"]}-{encoder["latent"]}-{encoder["dropout"]}-{encoder["head_count"]}-{encoder["batch_size"]}/best_model.pt --data_parallel --report_period 10 --early_stop 100 --lr {lr} --lr_gamma {lr_decay} --lr_step_size 15 --n_head {encoder["head_count"]} --layers {encoder["layer_count"]} --dropout {encoder["dropout"]} --d_latent {encoder["latent"]} --n_dec_blocks {dec_blocks} --dec_dropout {dec_dropout} --weight_decay {wd} --d_dec_ff {dec_ff} --batch_size {batch_size} --l_vocab 20 --language_only\n")
'''

for dec_blocks in [2]:
    for dec_dropout in [0.1]:
        for wd in [0.000001]:
            for dec_ff in [512]:
                for encoder in [large_encoder]:
                    #The scans model
                        #The noscans model

                    print(f'python rt_main.py --num_epochs 50 --predict --dec_freeze_type thaw_cross --decoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/lm_pretrain_test_3_19/best_lm_pre_model.pt --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/pos_predict_combined/{dec_blocks}-{dec_dropout}-{wd}-{dec_ff}-{encoder["kind"]} --encoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/{encoder["lr"]}-{encoder["lr_decay"]}-{encoder["layer_count"]}-{encoder["latent"]}-{encoder["dropout"]}-{encoder["head_count"]}-{encoder["batch_size"]}/best_model.pt --data_parallel --report_period 10 --early_stop 100 --lr {lr} --lr_gamma {lr_decay} --lr_step_size 15 --n_head {encoder["head_count"]} --layers {encoder["layer_count"]} --dropout {encoder["dropout"]} --d_latent {encoder["latent"]} --n_dec_blocks {dec_blocks} --dec_dropout {dec_dropout} --weight_decay {wd} --d_dec_ff {dec_ff} --batch_size {batch_size} --l_vocab 20\n')
