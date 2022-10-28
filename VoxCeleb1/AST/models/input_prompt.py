import torch
import torch.nn as nn
import numpy as np


class Input_prompt(nn.Module):
    """
    Reference:
    Bahng, Hyojin, et al. 
    "Visual Prompting: Modifying Pixel Space to Adapt Pre-trained Models." arXiv preprint arXiv:2203.17274 (2022).
    """
    def __init__(self, prompt_size, input_tdim, input_fdim):
        super(Model, self).__init__()
        self.base_fsize = input_fdim - prompt_size*2
        self.base_tsize = input_tdim - prompt_size*2
        self.pad_up = nn.Parameter(torch.randn([1, prompt_size, input_fdim]))
        self.pad_down = nn.Parameter(torch.randn([1, prompt_size, input_fdim]))
        self.pad_left = nn.Parameter(torch.randn([1, input_tdim - prompt_size*2, prompt_size]))
        self.pad_right = nn.Parameter(torch.randn([1, input_tdim - prompt_size*2, prompt_size]))

    def forward(self, x):

        base = torch.zeros(1, self.base_tsize, self.base_fsize).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=2)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=1)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt.to(x.device)
