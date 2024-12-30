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
from trs_to_pos import get_pos_seq
import json

class RT_Narrative_Data_Module(): #for random time splits

    def __init__(
        self, 
        task_list: List[str],
        batch_size: int = 16,
        transform: Optional[Callable] = None,
        num_workers: int = 2,
        train_val_test: tuple = (0.8, 0.1, 0.1), #
        segment_length: int = 10, #trs per segment
        delay: float = 4.5,
        tr_duration = 1.5,
        random_seed = 0
        ):
        
        assert(np.sum(train_val_test) == 1)

        self.task_list= task_list
        self.batch_size = batch_size             
        self.transform = transform 
        self.num_workers = num_workers
        self.train_val_test = train_val_test
        self.segment_length = segment_length
        self.delay = delay
        self.tr_duration = tr_duration

        #Setup task data Dict
        self.task_data = {}
        df = pd.read_csv('/home/wsm32/project/wsm_thesis_scratch/narratives/participants.tsv', sep='\t')
        with open("/home/wsm32/project/wsm_thesis_scratch/narratives/code/scan_exclude.json","r") as file:
            exclusion_dict = json.load(file)
        with open("/home/wsm32/project/wsm_thesis_scratch/narratives/code/fMRI-Embedding-narratives/duration.json","r") as file2:
            duration_dict = json.load(file2)

        rng = np.random.default_rng(seed)
        for task in task_list:
            contains_task = df['task'].apply(lambda x: self.task in x.split(','))
            to_exclude = df["participant_id"].isin(exclusion_dict[task].keys())
            valid_ids = df["participant_id"][contains_task & (~to_exclude)]
            num_subj = df["participant_id"][contains_task & (~to_exclude)].shape[0]

            num_segs = math.floor((duration_dict[task][1] - duration_dict[task][0]) / (tr_duration * segement_length))
            num_train = math.floor(train_val_test[0] * num_segs)
            num_val = math.floor(train_val_test[1] * num_segs)
            num_test = num_segs - num_train - num_val

            pattern_array = np.array([0]*num_train + [1]*num_val + [2]*num_test) # 0 for train, 1 for val, 2 for test
            rng.shuffle(pattern_array)

            pos_dir="/home/wsm32/project/wsm_thesis_scratch/narratives/stimuli/gentle/pos"
            file_path = join(pos_dir,task,"pos_align.json")
            with open(file_path, "r") as in_file:
                align_data = json.load(in_file)
            
            self.task_data[task] = {
                "valid_ids":valid_ids, 
                "num_subj" : num_subj,
                "start_time" : duration_dict[task][0],
                "end_time" : duration_dict[task][1],
                "stim_offset" : duration_dict[task][2],
                "num_train" : num_train,
                "num_val" : num_val,
                "num_test" : num_test,
                "pattern" : pattern_array,
                "align" : align_data
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
            train_loaders.append(RT_Narrative_Dataset(task, self.task_data[task], self.segement_length, self.train_val_test, self.delay, self.tr_duration, 0, transform=self.transform))
            val_loaders.append(RT_Narrative_Dataset(task, self.task_data[task], self.segement_length, self.train_val_test, self.delay, self.tr_duration, 1, transform=self.transform))
            test_loaders.append(RT_Narrative_Dataset(task,self.task_data[task], self.segement_length, self.train_val_test, self.delay, self.tr_duration, 2, transform=self.transform))
        
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
        train_val_test,
        delay,
        tr_duration,
        which_type,
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
        self.train_val_test = train_val_test
        self.delay = delay
        self.tr_duration = tr_duration
        self.which_type = which_type
        self.segs_per_sub = np.sum(task_data["pattern"] == which_type) # how many segments per subject (in this partition)
        self.num_ex =  self.segs_per_sub * task_data["num_subj"] # how many examples for this dataset
        self.these_seg_ids = np.where(task_data["pattern"] == which_type)

        self.signals = []
        for i in self.task_data["num_subj"]:
            s = get_vox(task, task_data["valid_ids"].iloc[i],"fsaverage6")
            signals.append(s)
        
    def seg_idx_to_trs(self, seg_idx): #takes a segment index, converts it to the relevant trs
        tr_offset = self.task_data["start_time"] / self.tr_duration
        return (seg_idx * self.segement_length + tr_offset, (seg_idx + 1) * self.segement_length + tr_offset)


        
    def __len__(self):
        return self.num_ex

    def __getitem__(self, i: int):
        
        subj_i = math.floor(i/self.segs_per_sub)
        seg_i = i % self.segs_per_sub
        tr_span = seg_idx_to_trs(self.these_seg_ids[seg_i])
        signal = self.signals[subj_i][tr_span[0]: tr_span[1], :]

        if self.transform is None:
            sample = signal
        else:
            sample = self.transform(signal)

        return torch.from_numpy(sample), get_pos_seq(self.task_data["align"],tr_span[0],tr_span[1],self.task_data["stim_offset"],tr_dur=self.tr_duration, delay=self.delay)