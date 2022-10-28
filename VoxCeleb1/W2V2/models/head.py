import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embd_dim, label_dim):
        super(Model, self).__init__()
        self.mlp_head1 = nn.Sequential(nn.Linear(embd_dim, 128))
        self.mlp_head2 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, label_dim))

    def forward(self, x, is_test = False):
        code = self.mlp_head1(x)
        if is_test: return code
        x = self.mlp_head2(code)
        return x