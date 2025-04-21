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
from os.path import join
from utils import pos_tags


class LM_Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.file = h5py.File(filename,'r')
        self.group = self.file["pos_vectors"]
        self.sorted_keys = sorted(self.group.keys(), key=lambda x: int(x.split('_')[1]))
        self.size = len(self.sorted_keys)
    def __len__(self):
        return self.size
    def __getitem__(self, i: int):
        key = self.sorted_keys[i]
        return self.group[key][()]

class LM_Data_Module(): #for random time splits
    def __init__(self, base_name, num_workers, batch_size):
        self.train_name = base_name + "_train.h5"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_name = base_name + "_validation.h5"
        self.test_name = base_name + "_test.h5"

    def _setup_dataloaders(self):

        train_data, val_data, test_data = self._create_datasets()
        pf = 2 if self.num_workers > 0 else None


        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=pf
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,            
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=pf
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=pf
        )

        return train_loader, val_loader, test_loader

    def _create_datasets(self):
        train = LM_Dataset(self.train_name)
        val = LM_Dataset(self.val_name)
        test = LM_Dataset(self.test_name)
        
        return train, val, test
