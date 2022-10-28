import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class Model(nn.Module):
    def __init__(self, embd_dim, label_dim):
        super(Model, self).__init__()
        self.mlp_head = nn.Sequential(nn.LayerNorm(embd_dim), nn.Linear(embd_dim, label_dim))

    def forward(self, x):
        x = self.mlp_head(x)
        return x
