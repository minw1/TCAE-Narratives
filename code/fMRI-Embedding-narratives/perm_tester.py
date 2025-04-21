import json
import subprocess
import re
import os

batch_size = 16
lr = 0.001
lr_decay = 0.5

small_encoder = {
    "layer_count": 1,
    "latent": 64,
    "head_count": 4,
    "dropout": 0.05,
    "batch_size": 16,
    "kind": "small",
}

large_encoder = {
    "layer_count": 2,
    "latent": 128,
    "head_count": 8,
    "dropout": 0.05,
    "batch_size": 16,
    "kind": "large",
}

dec_dropout = 0.1

def extract_json_from_output(output: str) -> str:
    pattern = r"PERM_BEGIN\s*(.*?)\s*PERM_END"
    match = re.search(pattern, output, re.DOTALL)
    if not match:
        raise ValueError("Markers PERM_BEGIN and PERM_END not found.")
    return match.group(1)

perm_output_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/combined_run/perm_testing"
os.makedirs(perm_output_dir, exist_ok=True)

for dec_blocks in [1, 2, 4]:
    for n_dec_head in [4]:
        for wd in [0.000001, 0.00001]:
            wd = f"{wd:.6f}"
            for dec_ff in [512, 1024]:
                for encoder in [small_encoder, large_encoder]:
                    for lr in [0.001, 0.0001]:
                        for df in ["thaw_cross", "none"]:
                            for enf in ["all", "none"]:
                                dec_filename = f"{dec_blocks}-{n_dec_head}-{wd}-{dec_ff}-{encoder['kind']}-{lr}"
                                filename = f"{dec_blocks}-{n_dec_head}-{wd}-{dec_ff}-{encoder['kind']}-{lr}-{df}-{enf}"
                                enc_filename = f"/home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/0.001-0.5-{encoder['layer_count']}-{encoder['latent']}-0.05-{encoder['head_count']}-16/best_model.pt"
                                output_file = os.path.join(perm_output_dir, f"{filename}.txt")

                                setup = [
                                    "source /gpfs/gibbs/project/frank/wsm32/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/tcae_new/bin/activate",
                                    "module load CUDA",
                                    "module load cuDNN"
                                ]

                                shell_command = f"""
                                python rt_main.py --num_epochs 50 --predict --perm --data_parallel \
                                --checkpoint /home/wsm32/project/wsm_thesis_scratch/narratives/combined_run/predictors/{filename}/best_predictor_model.pt \
                                --dec_freeze_type {df} --enc_freeze_type {enf} \
                                --decoder_base /home/wsm32/project/wsm_thesis_scratch/narratives/combined_run/decoder_bases/{dec_filename}/best_lm_pre_model.pt \
                                --exp_dir /home/wsm32/project/wsm_thesis_scratch/narratives/ \
                                --encoder_base {enc_filename} --report_period 10 --early_stop 100 --lr {lr} \
                                --lr_gamma {lr_decay} --lr_step_size 15 --n_head {encoder['head_count']} \
                                --layers {encoder['layer_count']} --dropout 0.05 --d_latent {encoder['latent']} \
                                --n_dec_blocks {dec_blocks} --dec_dropout {dec_dropout} --weight_decay {wd} \
                                --d_dec_ff {dec_ff} --n_dec_head {n_dec_head} --batch_size 16 --l_vocab 20
                                """

                                full_command = " && ".join(setup + [shell_command.strip()])

                                try:
                                    print("running command:")
                                    result = subprocess.run(
                                        ["bash", "-c", full_command],
                                        capture_output=True,
                                        text=True
                                    )

                                    with open(output_file, "w") as fout:
                                        fout.write(result.stdout)
                                        fout.write("\n--- STDERR ---\n")
                                        fout.write(result.stderr)
                                    print("saved")
                                except Exception as e:
                                    print(f"FAILED {filename}: {e}")
