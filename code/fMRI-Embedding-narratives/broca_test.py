from create_maps import *
import numpy as np
from functools import reduce
import nibabel as nib
import numpy as np
from nilearn import plotting
import os

lh_f = "/home/wsm32/project/wsm_thesis_scratch/narratives/fs6_transfer/label/lh.aparc.a2009s.annot"
rh_f = "/home/wsm32/project/wsm_thesis_scratch/narratives/fs6_transfer/label/rh.aparc.a2009s.annot"
region_dict = load_annot_to_binary_dict(lh_f, rh_f)
brocas = ["L_G_front_inf-Opercular", "L_G_front_inf-Triangul"]
#use a boolean OR to combine all regions in brocas
broca_map = reduce(np.bitwise_or, [region_dict[region] for region in brocas])
print(np.sum(broca_map))
print(broca_map.shape)

# Load the fsaverage6 surface
fsaverage_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/fs6_transfer"

lh_pial = os.path.join(fsaverage_dir, "surf", "lh.pial")
rh_pial = os.path.join(fsaverage_dir, "surf", "rh.pial")

# Your data array
# Replace this with your actual data, ensure it's the right length
data_l = broca_map[0:40962] 

# Load surfaces
lh_coords, lh_faces = nib.freesurfer.read_geometry(lh_pial)
rh_coords, rh_faces = nib.freesurfer.read_geometry(rh_pial)

# Visualize left hemisphere
view = plotting.view_surf((lh_coords, lh_faces), surf_map=data_l, cmap='viridis')
view.save_as_html("output.html")
