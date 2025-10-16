import torch.nn as nn


class ScalarWrapper(nn.Module):
    def __init__(self, model, target):
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, x):
        y = self.model(x)
        s = self.target(y)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return s
