import sys 
import numpy as np 

sys.path.append('/../../sequence_models')
from sequence_models.structure import Attention1d
from utils import vocab, pad_index

import torch
import torch.nn as nn

class ESMAttention1d(nn.Module):
    """Outputs of the ESM model with the attention1d"""
    def __init__(self, max_length, d_embedding): # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(ESMAttention1d, self).__init__()
        self.attention1d = Attention1d(in_dim=d_embedding) # ???
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)
    
    def forward(self, x, input_mask):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x

class ESMAttention1dMean(nn.Module):
    """Attention1d removed, leaving a basic linear model"""
    def __init__(self, d_embedding): # [batch x embedding (1280)]  --> [batch x 1]
        super(ESMAttention1dMean, self).__init__()
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x
