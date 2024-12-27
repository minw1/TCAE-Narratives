import os
import h5py, logging, pickle
from pathlib import Path
from typing import Callable, Optional, Union
import random
import torch
import numpy as np


class Volume_Data_Module():

    def __init__(
        self, 
        data_path: Path,
        batch_size: int = 16,
        sample_rate: float = 1.0,
        transform: Optional[Callable] = None,
        num_workers: int = 2
        ):

        self.root = data_path
        self.batch_size = batch_size        
        self.sample_rate = sample_rate        
        self.transform = transform 
        self.num_workers = num_workers

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


        train_data = Volume_Dataset(self.root+'train',self.sample_rate, transform=self.transform)
        val_data = Volume_Dataset(self.root+'val',self.sample_rate, transform=self.transform)
        test_data = Volume_Dataset(self.root,self.sample_rate, transform=self.transform) 
        
        return train_data, val_data, test_data


class Volume_Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to fMRI volume data.
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        sample_rate: Optional[float] = None,
        use_dataset_cache: bool = True,
        dataset_cache_file: Union[str, Path, os.PathLike] = "volume_dataset_cache.pkl",
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

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        
        self.examples = []

        # set default sampling rate if none given
        if sample_rate is None:
            sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                self.examples += [fname]
                
               
            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)

        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by vertices
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):

        fname = self.examples[i]

        with h5py.File(fname, "r") as hf:
            signal = hf['signal'][()]

        if self.transform is None:
            sample = signal
        else:
            sample = self.transform(signal)

        return torch.from_numpy(sample)


class HCP_Volume_Data_Module():

    def __init__(
        self, 
        data_path: Path,
        task: str = 'EMOTION',
        sample_rate: float = 1.0,
        transform: Optional[Callable] = None,
        batch_size: int = 16,
        num_workers: int = 2
        ):

        self.root = data_path
        self.task = task
        self.sample_rate = sample_rate        
        self.transform = transform 
        self.batch_size = batch_size
        self.num_workers = num_workers

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


        train_data = HCP_Volume_Dataset(self.root+'train',self.root+'task_designs/',self.task,self.sample_rate, transform=self.transform)
        val_data = HCP_Volume_Dataset(self.root+'val',self.root+'task_designs/',self.task,self.sample_rate, transform=self.transform)
        test_data = HCP_Volume_Dataset(self.root+'test',self.root+'task_designs/',self.task,self.sample_rate, transform=self.transform) 
        
        return train_data, val_data, test_data


class HCP_Volume_Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to HCP fMRI volume data.
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        td_root: Union[str, Path, os.PathLike],
        task: str = 'EMOTION',
        sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "HCP_volume_dataset_cache.pkl",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            task: 'EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM'.
            td_root: Path to the task designs.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction of the slices should be loaded. 
                         Defaults to 1 if no value is given.
            use_dataset_cache: Whether to cache dataset metadata.
            dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
            transform: Optional; A callable object that transform the raw data into appropriate form. 
                       The transform function should take 'signal' as inputs.
        """
        self.task = task
        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        
        self.examples = []
        self.task_designs = []

        # set default sampling rate if none given
        if sample_rate is None:
            sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                self.examples += [fname]
                self.task_designs += [td_root+str(fname)[-9:-3]+'_'+self.task+'.h5']
               
            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                dataset_cache[root+'_task_design'] = self.task_designs
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)

        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]
            self.task_designs = dataset_cache[root+'_task_design']

        # subsample if desired
        if sample_rate < 1.0:  # sample by vertices
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            self.task_designs = self.task_designs[:num_examples]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):

        fname = self.examples[i]
        # print(i)
        # print(str(fname)[-9:-3])
        # print('***********')
        
        label_fname = self.task_designs[i]

        with h5py.File(fname, "r") as hf:
            signal = hf[self.task][()]
            # print(signal.shape)

        with h5py.File(label_fname, "r") as hf:
            label = hf['label'][()]
            
        if self.transform is None:
            sample = signal
        else:
            sample = self.transform(signal)

        return torch.from_numpy(sample),torch.from_numpy(label)