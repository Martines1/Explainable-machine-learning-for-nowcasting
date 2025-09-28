import torch

class RegressionTarget:
    def __init__(self, mode, k_value=None):
        self.mode = mode.lower()
        if self.mode == "topk":
            self.k_value = k_value

    def __call__(self, y):
        match self.mode:
            case "mean":
                return y.mean()
            case "max":
                return y.max()
            case "topk":
                k = max(1, int(self.k_value * y.numel()))
                vals, _ = torch.topk(y.reshape(-1), k)
                return vals.mean()
        return y.max()

