import h5py
import numpy as np
from gen_vox import get_vox
import pandas as pd
import threading
import json
from os.path import join

out_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/h5/"
res = 81924

def create_h5(task):
    df = pd.read_csv('/home/wsm32/project/wsm_thesis_scratch/narratives/participants.tsv', sep='\t')
    with open("/home/wsm32/project/wsm_thesis_scratch/narratives/code/scan_exclude.json","r") as file:
        exclusion_dict = json.load(file)


    contains_task = df['task'].apply(lambda x: task in x.split(','))
    to_exclude = df["participant_id"].isin(exclusion_dict[task].keys())
    valid_ids = df["participant_id"][contains_task & (~to_exclude)]
    num_subj = df["participant_id"][contains_task & (~to_exclude)].shape[0]

    #for i in range(num_subj):
    for i in range(1):
        s = get_vox(task, valid_ids.iloc[i],"fsaverage6")
        with h5py.File(join(out_dir, f"{task}_{valid_ids.iloc[i]}.hdf5"), "x", rdcc_nslots=5000, rdcc_nbytes=100000000) as f:
            dset = f.create_dataset("data", data=s, chunks=(20,res))

with h5py.File(join(out_dir, f"{"pieman"}_{"sub-002"}.hdf5"), "r", rdcc_nslots=5000, rdcc_nbytes=100000000) as f:
    s = get_vox("pieman","sub-002","fsaverage6")
    j = f['data'][:]
    print(np.mean(s == j))