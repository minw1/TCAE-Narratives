from os.path import join
import json
import numpy as np
from gifti_io import read_gifti


pieman_doubleruns = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-008", "sub-010", "sub-011", "sub-012", "sub-013", "sub-014", "sub-015", "sub-016"]

def get_vox(task, subject, space):
    
    base_dir = '/home/wsm32/palmer_scratch/wsm_thesis_scratch/narratives/'
    clean_dir = join(base_dir, 'derivatives', 'afni-smooth', subject, 'func')
    run = ""
    if task == "pieman" and subject in pieman_doubleruns:
        run = "_run-1"
    

    clean_fn_L  = join(clean_dir, (f'{subject}_task-{task}{run}_space-{space}_'
                             f'hemi-{"L"}_desc-clean.func.gii'))
    clean_fn_R = join(clean_dir, (f'{subject}_task-{task}{run}_space-{space}_'
                             f'hemi-{"R"}_desc-clean.func.gii'))

    clean_map_L = read_gifti(clean_fn_L)
    clean_map_R = read_gifti(clean_fn_R)


    return np.hstack((clean_map_L, clean_map_R))


