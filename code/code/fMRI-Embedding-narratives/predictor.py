import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Brain_State_Predictor(nn.Module):
    """
    Predict the brain state with fMRI embedding.
    """
    def __init__(
        self, 
        num_class: int = 3,
        in_planes: int = 256,
        mid_planes: int = 32
        ): 
        """
        Args:
            num_class (int): Number of classes in classification.

        """
        super(Brain_State_Predictor, self).__init__()

        self.fc1 = nn.Linear(in_planes, mid_planes)
        self.fc2 = nn.Linear(mid_planes, num_class)

        self.relu = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x