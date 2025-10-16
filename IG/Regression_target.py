import torch


class RegressionTargetIG:
    def __init__(self, mode, k_value=None):
        self.mode = mode.lower()
        if self.mode == "topk":
            self.k_value = k_value

    def __call__(self, y):
        if y.dim() == 4:
            y = y.squeeze(1)
        B, H, W = y.shape
        match self.mode:
            case "mean":
                return y.mean(dim=(1, 2), keepdim=True)
            case "max":
                return y.amax(dim=(1, 2), keepdim=True)
            case "topk":
                k = max(1, int(self.k_value * y.numel()))
                vals, _ = torch.topk(y.reshape(B, -1), k, dim=1)
                return vals.mean(dim=1, keepdim=True)
        return y.max(dim=1, keepdim=True)
