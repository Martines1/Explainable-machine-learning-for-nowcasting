import sys

from perturbation import difference
from perturbation.loss_functions import *
import torch


class Perturbation:
    def __init__(self, model, input_, device, ground_truth=None):
        # input_ shape should be (B, C, H, W) or (B, C, W, H)
        self.model = model
        self.model.eval()
        self.input = input_
        self.ground_truth = ground_truth
        self.device = device

    def forward(self, x):
        with torch.inference_mode():
            y_t = self.model(x)
        return y_t.detach().cpu().numpy()[0, 0, :, :]

    def turn_off_channels(self, ch, baseline, loss="logcosh"):
        if type(ch) is not list:
            sys.exit("Channels parameter must be list")
        if len(ch) == 0:
            return

        input_pert = self.input.detach().cpu().numpy().copy()
        for c in ch:
            input_pert[:, c, :, :] = baseline
        input_pert = torch.from_numpy(input_pert).to(self.device)
        loss_f = get_function(loss)
        base_pred = self.forward(self.input)
        pert_pred = self.forward(input_pert)
        loss_result = loss_f.calculate(pert_pred, base_pred)
        result = {"diff": loss_result, "base_pred": base_pred, "pert_pred": pert_pred}
        if self.ground_truth is not None:
            gt_loss_result = loss_f.calculate(pert_pred, self.ground_truth)
            result["gt_perp_diff"] = gt_loss_result
            gt_base_diff = loss_f.calculate(base_pred, self.ground_truth)
            result["gt_base_diff"] = gt_base_diff
        return result

    def perturbate_channels(self, channels, baseline, loss="logcosh", window=32):
        base_input = self.input.detach().cpu().numpy()
        b, c, h, w = base_input.shape

        loss_f = get_function(loss)
        base_pred = self.forward(self.input)

        if self.ground_truth is not None:
            gt_base_diff = loss_f.calculate(base_pred, self.ground_truth)

        masks = difference.calculate_diff_unique(base_input[0], np.log(0.01))
        importance = np.zeros((c, h, w), dtype=np.float32)
        for ch in channels:
            frame = base_input[0, ch, :, :].copy()
            diff_mask = None
            for y in range(0, h, window):
                y2 = min(y + window, h)
                for x in range(0, w, window):
                    x2 = min(x + window, w)
                    win_mask = diff_mask[y:y2, x:x2]
                    if not win_mask.any():
                        continue

                    pert_input = base_input.copy()
                    patch = pert_input[0, ch, y:y2, x:x2]
                    patch_baseline = baseline[y:y2, x:x2]
                    patch[win_mask] = patch_baseline[win_mask]
