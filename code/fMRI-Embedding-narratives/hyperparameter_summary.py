import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def get_validation_loss(summary_dir):
    """
    Extracts the minimum validation loss from a TensorBoard event file.
    """
    event_acc = EventAccumulator(summary_dir)
    event_acc.Reload()

    # Assume 'validation_loss' is the tag used; replace it with the actual tag name.
    tag = 'Val_Loss'
    if tag not in event_acc.Tags()['scalars']:
        return None  # Validation loss not found in this summary.

    scalars = event_acc.Scalars(tag)
    # Get the minimum validation loss
    min_val_loss = min(scalar.value for scalar in scalars)
    return min_val_loss

def find_top_models(base_dir):
    """
    Finds the top N models with the best validation loss.
    """
    model_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    results = {}

    for model_dir in model_dirs:
        summary_dir = os.path.join(model_dir, "summary")
        if os.path.exists(summary_dir):
            val_loss = get_validation_loss(summary_dir)
            if val_loss is not None:
                results[model_dir] = val_loss

    # Sort results by validation loss (ascending) and return the top N
    sorted_results = sorted(results.items(), key=lambda item: item[1])
    return sorted_results

# Directory containing all model directories
base_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/rt_large"

top_models = find_top_models(base_dir)


def get_hps(dir_str):
    toks = dir_str.split("/")
    nums = toks[-1].split("-")
    return tuple(nums)

param_vals = {}
for k,v in top_models:
    param_vals[get_hps(k)] = v

lrs = [0.05, 0.01, 0.001]
lr_decays = [0.9, 0.5, 0.1]
layer_counts = [1, 2]
latents = [64, 128]
dropouts = [0.05]
head_counts = [4,8]
batch_sizes = [16, 128]


for lr in lrs:
    those = {k: v for k, v in param_vals.items() if float(k[0]) == lr}
    print(f"Mean for lr={lr}: {np.mean(list(those.values()))}")

for lr_decay in lr_decays:
    those = {k: v for k, v in param_vals.items() if float(k[1]) == lr_decay}
    print(f"Mean for lr_decay={lr_decay}: {np.mean(list(those.values()))}")

for layer_count in layer_counts:
    those = {k: v for k, v in param_vals.items() if float(k[2]) == layer_count}
    print(f"Mean for layer_count={layer_count}: {np.mean(list(those.values()))}")

for latent in latents:
    those = {k: v for k, v in param_vals.items() if float(k[3]) == latent}
    print(f"Mean for latent={latent}: {np.mean(list(those.values()))}")

for head_count in head_counts:
    those = {k: v for k, v in param_vals.items() if float(k[5]) == head_count}
    print(f"Mean for head_count={head_count}: {np.mean(list(those.values()))}")
for batch_size in batch_sizes:
    those = {k: v for k, v in param_vals.items() if float(k[6]) == batch_size}
    print(f"Mean for batch size={batch_size}: {np.mean(list(those.values()))}")


for i in range(144):
    print(top_models[i])