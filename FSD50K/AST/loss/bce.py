import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        
        self.criterion  = nn.BCEWithLogitsLoss()

    def forward(self, x, label=None):

        nloss   = self.criterion(x, label)
        return nloss