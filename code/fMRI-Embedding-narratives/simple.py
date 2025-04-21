from trs_to_pos import get_pos_seq
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


task = "notthefallintact"
pos_dir="/home/wsm32/project/wsm_thesis_scratch/narratives/stimuli/gentle/pos"
file_path = join(pos_dir,task,"pos_align.json")
with open(file_path, "r") as in_file:
    align_data = json.load(in_file)

pos_seq = get_pos_seq(align_data, 239, 249 ,4.5 ,1.5, delay=4.5)

#print(pos_seq)

print(torch.nn.Transformer.generate_square_subsequent_mask(5))