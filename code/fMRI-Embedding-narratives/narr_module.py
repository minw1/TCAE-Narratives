import os
import h5py, logging, pickle
from pathlib import Path
from typing import Callable, Optional, Union
import random
import torch
import math
import numpy as np
import pandas as pd
from gen_vox import get_vox
import json

class CT_Narrative_Data_Module(): #for consecutive time splits

    def __init__(
        self, 
        task: str,
        task_tr: int,
        batch_size: int = 16,
        transform: Optional[Callable] = None,
        num_workers: int = 2,
        train_val_test: tuple = (0.8, 0.1, 0.1),
        frames_per: int = 10
        ):
        
        assert(np.sum(train_val_test) == 1)
        
        self.batch_size = batch_size             
        self.transform = transform 
        self.num_workers = num_workers
        self.train_val_test = train_val_test
        self.task = task
        self.task_tr = task_tr
        self.frames_per = frames_per

    def _setup_dataloaders(self):

        train_data, val_data, test_data = self._create_datasets()

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,            
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

        return train_loader, val_loader, test_loader

    def _create_datasets(self):


        train_data = CT_Narrative_Dataset(self.task, self.frames_per, 0, self.task_tr*self.train_val_test[0],\
                                          transform=self.transform)
        val_data = CT_Narrative_Dataset(self.task, self.frames_per, self.task_tr*self.train_val_test[0],\
                                        self.task_tr*(self.train_val_test[0] + self.train_val_test[1]), transform=self.transform)
        test_data = CT_Narrative_Dataset(self.task, self.frames_per, \
                                         self.task_tr*(self.train_val_test[0] + self.train_val_test[1]), self.task_tr,\
                                         transform=self.transform)
        
        return train_data, val_data, test_data


class CT_Narrative_Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to fMRI volume data.
    """
    def __init__(
        self,
        task,
        trs_per_shot,
        start_tr,
        end_tr,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction of the slices should be loaded. 
                         Defaults to 1 if no value is given.
            use_dataset_cache: Whether to cache dataset metadata.
            dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
            transform: Optional; A callable object that transform the raw data into appropriate form. 
                       The transform function should take 'signal' as inputs.
        """
        self.task = task
        self.transform = transform
        self.trs_per_shot = trs_per_shot
        self.shots_per_subj = math.floor((end_tr - start_tr)/trs_per_shot)
        
        df = pd.read_csv('/home/wsm32/palmer_scratch/wsm_thesis_scratch/narratives/participants.tsv', sep='\t')
        
        with open("/home/wsm32/palmer_scratch/wsm_thesis_scratch/narratives/code/scan_exclude.json","r") as file:
            exclusion_dict = json.load(file)
        contains_task = df['task'].apply(lambda x: self.task in x.split(','))
        to_exclude = df["participant_id"].isin(exclusion_dict[task].keys())
        
        self.valid_ids = df["participant_id"][contains_task & (~to_exclude)]
        self.num_subj = df["participant_id"][contains_task & (~to_exclude)].shape[0]
        self.num_ex = self.shots_per_subj * self.num_subj

        
    def __len__(self):
        return self.num_ex

    def __getitem__(self, i: int):
        
        subj_i = math.floor(i/self.shots_per_subj)
        shot_i = i % self.shots_per_subj
        
        signal = get_vox(self.task,self.valid_ids.iloc[subj_i],"fsaverage6")[shot_i*self.trs_per_shot: \
                                                                            (shot_i + 1)*self.trs_per_shot,:]

        if self.transform is None:
            sample = signal
        else:
            sample = self.transform(signal)

        return torch.from_numpy(sample), torch.tensor(1) # a dummy label for now
