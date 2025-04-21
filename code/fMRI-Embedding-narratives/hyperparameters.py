

lrs = [0.05, 0.01, 0.001]
lr_decays = [0.9, 0.5, 0.1]
layer_counts = [1, 2]
latents = [64, 128]
dropouts = [0.05]
head_counts = [4,8]
batch_sizes = [16, 128]



with open("hyperjobs.txt", "w") as file:
    for lr in lrs:
        for lr_decay in lr_decays:
            for layer_count in layer_counts:
                for latent in latents:
                    for dropout in dropouts:
                        for head_count in head_counts:
                            for batch_size in batch_sizes:
                                file.write("module load CUDA; ")
                                file.write("module load cuDNN; ")
                                file.write("source activate tcae; ")
                                file.write(f"python rt_main.py --num_epochs 100 --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/{lr}-{lr_decay}-{layer_count}-{latent}-{dropout}-{head_count}-{batch_size} --data_parallel --report_period 10 --early_stop 100 --lr {lr} --lr_gamma {lr_decay} --lr_step_size 15 --n_head {head_count} --layers {layer_count} --dropout {dropout} --d_latent {latent} --batch_size {batch_size}\n")