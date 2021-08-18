import sys 
import numpy as np 

sys.path.append('/home/v-jodymou/sequence_models')
from sequence_models.convolutional import ByteNet
from sequence_models.structure import Attention1d
from utils import vocab, pad_index

sys.path.append('/home/v-jodymou/esm')
import esm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



class Linear_Base(nn.Module):
    """Linear model used with both logistic and linear regression"""
    def __init__(self, max_length):
        super(Linear_Base, self).__init__()
        self.max_length = max_length
        self.embed_dim = len(vocab) + 1
        self.lut = torch.tensor(np.eye(len(vocab) + 1), dtype=float) # look up table for one-hot encoding
        self.embedding = nn.Embedding.from_pretrained(self.lut, padding_idx=pad_index, freeze=True)
        #self.linear = nn.Linear(self.max_length * self.embed_dim, self.max_length * self.embed_dim)
        #self.relu = nn.ReLU()
        self.out = nn.Linear(self.max_length * self.embed_dim, 1)

    
    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.reshape(x, (-1, self.max_length * self.embed_dim))
        #x = self.linear(x.float())
        #x = self.relu(x)
        x = self.out(x.float())
        return x

class LeastSquares(nn.Module):
    def __init__(self, max_length):
        super(LeastSquares, self).__init__()
        self.embed_dim = len(vocab) + 1
        self.max_length = max_length
        self.lut = torch.tensor(np.eye(len(vocab) + 1), dtype=float) # look up table for one-hot encoding
        self.embedding = nn.Embedding.from_pretrained(self.lut, padding_idx=pad_index, freeze=True)
        
    def forward(self, x, y, training): 
        x = self.embedding(x.long())
        x = torch.reshape(x, (-1, self.max_length * self.embed_dim))
        if training: 
            return torch.linalg.lstsq(x.float(), y.unsqueeze(-1).float()) 
        else:
            return x

class ByteAttention1d(nn.Module):
    """ByteNet + Attention 1d + Linear"""
    # [batch x length ] --> ByteNetLM --> [batch x length x embed] 
    # --> Attention1d --> [batch x embed] --> Linear/Logits --> [batch x 1] 
    def __init__(self, d_embedding, d_model, n_layers, kernel_size, r, activation, max_length):
        super(ByteAttention1d, self).__init__()
        self.max_length = max_length
        self.bytenet = ByteNet(n_tokens = max_length, d_embedding = d_embedding, d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r, padding_idx = pad_index, activation = activation)
        self.attention1d = Attention1d(in_dim = d_model)
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x, input_mask): 
        x = self.bytenet(x.long(), input_mask=input_mask)
        x = self.attention1d(x, input_mask=input_mask)
        x = self.linear(x)
        return x

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
