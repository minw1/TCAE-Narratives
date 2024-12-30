import os
import h5py, logging, pickle
from pathlib import Path
from typing import Callable, Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import random
import torch
import math
import numpy as np
import pandas as pd
from gen_vox import get_vox
import json

class RT_Narrative_Data_Module(): #for random (not exactly random, more cyclical) time splits

    def __init__(
        self, 
        task_list: List[str],
        batch_size: int = 16,
        transform: Optional[Callable] = None,
        num_workers: int = 2,
        train_val_test: tuple = (8, 1, 1), #In a cycle of 10 segments, how many of those segments should be train, val, test.
        segment_length: int = 10 #trs per segment
        ):
        
        assert(np.sum(train_val_test) == 10)
        
        self.batch_size = batch_size             
        self.transform = transform 
        self.num_workers = num_workers
        self.train_val_test = train_val_test
        self.task_list= task_list
        self.segment_length = segment_length

        #Setup task data Dict
        self.task_data = {}
        df = pd.read_csv('/home/wsm32/project/wsm_thesis_scratch/narratives/participants.tsv', sep='\t')
        with open("/home/wsm32/project/wsm_thesis_scratch/narratives/code/scan_exclude.json","r") as file:
            exclusion_dict = json.load(file)
        with open("/home/wsm32/project/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/duration.json","r") as file2:
            duration_dict = json.load(file2)
            
        for task in task_list:
            contains_task = df['task'].apply(lambda x: self.task in x.split(','))
            to_exclude = df["participant_id"].isin(exclusion_dict[task].keys())
            valid_ids = df["participant_id"][contains_task & (~to_exclude)]
            num_subj = df["participant_id"][contains_task & (~to_exclude)].shape[0]
            self.task_data[task] = {
                "valid_ids":valid_ids, 
                "num_subj" = num_subj,
                "start_time" = duration_dict[task][0],
                "end_time" = duration_dict[task][1]
            }
        


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
        train_loaders = []
        val_loaders = []
        test_loaders = []

        for task in self.task_list:
            train_loaders.append(RT_Narrative_Dataset(task, self.task_data[task], self.segement_length, cycle = (0,train_val_test[0]), transform=self.transform))
            val_loaders.append(RT_Narrative_Dataset(task, self.task_data[task], self.segement_length, cycle = (train_val_test[0],train_val_test[0]+train_val_test[1]), transform=self.transform))
            test_loaders.append(RT_Narrative_Dataset(task,self.task_data[task], self.segement_length, cycle = (train_val_test[0]+train_val_test[1], 10), transform=self.transform))
        
        return ConcatDataset(train_loaders), ConcatDataset(val_loaders), ConcatDataset(test_loaders)

        

class RT_Narrative_Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to fMRI volume data.
    """
    def __init__(
        self,
        task,
        task_data,
        segment_length,
        cycle,
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
        self.task_data = task_data
        self.transform = transform
        self.segment_length = segement_length
        self.cycle = cycle
        self.valid_ids = task_data["valid_ids"]
        self.num_subj = task_data["num_subj"]
        self.start_time = task_data["start_time"]
        self.end_time = task_data["end_time"]

        self.num_ex = #FIGURE OUT

        
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