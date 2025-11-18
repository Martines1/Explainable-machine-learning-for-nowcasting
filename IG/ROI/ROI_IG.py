import torch
import torch.nn as nn


class RegressionTargetROI(nn.Module):
    def __init__(self, roi_mask, mode="mean"):
        super().__init__()
        self.mode = mode

        if roi_mask.ndim == 2:
            roi_mask = roi_mask[None, None, ...]
        self.register_buffer("mask", roi_mask.float())

    def forward(self, y):
        masked = y * self.mask
        if self.mode == "sum":
            return masked.sum(dim=(1, 2, 3), keepdim=True)
        elif self.mode == "mean":
            denom = self.mask.sum(dim=(1, 2, 3), keepdim=True)
            return masked.sum(dim=(1, 2, 3), keepdim=True) / denom
        elif self.mode == "max":
            vals = masked[self.mask > 0]
            return vals.max().unsqueeze(0).unsqueeze(0)
