from create_maps import *
import numpy as np
from functools import reduce
import nibabel as nib
import numpy as np
from nilearn import plotting
import os
import torch

def zscore(x):
    std = np.std(x)
    mean = np.mean(x)
    return ((x - mean)/std)

fsaverage_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/fs6_transfer"
lh_pial = os.path.join(fsaverage_dir, "surf", "lh.pial")
rh_pial = os.path.join(fsaverage_dir, "surf", "rh.pial")

large_baseline = "/home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/0.001-0.5-2-128-0.05-8-16/best_model.pt"
small_baseline = "/home/wsm32/project/wsm_thesis_scratch/narratives/rt_large/0.001-0.5-1-64-0.05-4-16/best_model.pt"


model_file = "/home/wsm32/project/wsm_thesis_scratch/narratives/pos_predict_combined_3_29/2-0.1-1e-06-512-large/best_predictor_model.pt"

state_dict = torch.load(model_file, map_location=torch.device('cpu'))['model']
state_dict_baseline = torch.load(model_file_baseline, map_location=torch.device('cpu'))['model']

print(state_dict.keys())
print(state_dict["module.encoder.en_emd.weight"].shape)

lin_embedding = state_dict["module.encoder.en_emd.weight"].numpy()
lin_mean_embedding = np.mean(lin_embedding, axis=0)
print(lin_mean_embedding.shape)


lin_embedding_baseline = state_dict_baseline["module.encoder.en_emd.weight"].numpy()
lin_mean_embedding_baseline = np.mean(lin_embedding_baseline, axis=0)

data_l = zscore(lin_mean_embedding[0:40962] - lin_mean_embedding_baseline[0:40962])
data_r = zscore(lin_mean_embedding[40962:] - lin_mean_embedding_baseline[40962:])



lh_coords, lh_faces = nib.freesurfer.read_geometry(lh_pial)
rh_coords, rh_faces = nib.freesurfer.read_geometry(rh_pial)

# Visualize left hemisphere
view = plotting.view_surf((lh_coords, lh_faces), surf_map=data_l, cmap='viridis')
view.save_as_html("delta_output.html")

view = plotting.view_surf((rh_coords, rh_faces), surf_map=data_r, cmap='viridis')
view.save_as_html("delta_output_r.html")