import sys

from matplotlib import pyplot as plt

from perturbation import difference
from perturbation.loss_functions import *
import torch


class Perturbation:
    def __init__(self, model, input_, device, ground_truth):
        # input_ shape should be (B, C, H, W) or (B, C, W, H)
        self.importance = None
        self.model = model
        self.model.eval()
        self.input = input_
        self.ground_truth = ground_truth
        self.device = device

    def forward(self, x):
        with torch.inference_mode():
            y_t = self.model(x)
        return y_t.detach().cpu().numpy()[0, 0, :, :]

    def turn_off_channels(self, ch, baseline, loss="logcosh", rain_value=None):
        if type(ch) is not list:
            sys.exit("Channels parameter must be list")
        if len(ch) == 0:
            return
        rain_value = np.round(rain_value, 3)
        input_pert = self.input.detach().cpu().numpy().copy()
        for c in ch:
            input_pert[:, c, :, :] = baseline
        input_pert = torch.from_numpy(input_pert).to(self.device)
        loss_f = get_function(loss)
        base_pred = self.forward(self.input)
        pert_pred = self.forward(input_pert)
        result = dict()
        result["pert_pred"] = pert_pred
        gt_loss_result = loss_f.calculate(pert_pred, self.ground_truth, rain_value)
        result["gt_perp_diff"] = gt_loss_result
        gt_base_diff = loss_f.calculate(base_pred, self.ground_truth, rain_value)
        result["gt_base_diff"] = gt_base_diff
        return result

    def perturbate_channels(self, baseline, masks, loss="logcosh", window=128):
        base_input = self.input.detach().cpu().numpy()
        b, c, h, w = base_input.shape

        prediction_offset = 5
        loss_f = get_function(loss)
        base_pred = self.forward(self.input)

        counter = self.get_counter(base_input, masks, window)

        importance = np.zeros((c, h, w), dtype=np.float32)
        for ch in range(1):
            diff_mask = masks[ch]
            internal_counter = 0
            for y in range(0, h, window):
                y2 = min(y + window, h)
                for x in range(0, w, window):
                    x2 = min(x + window, w)
                    win_mask = diff_mask[y:y2, x:x2]
                    if not win_mask.any():
                        continue
                    mask_count = np.sum(win_mask)
                    gt_mask = self.ground_truth[y:y2, x:x2]
                    base_pred_mask = base_pred[y:y2, x:x2]
                    internal_counter += 1
                    print(f'Processing channel {ch}, done {(internal_counter / counter[ch]) * 100:.2f} %')
                    input_copy = base_input.copy()
                    frame = input_copy[0, ch, :, :]
                    patch = frame[y:y2, x:x2]
                    patch[win_mask] = baseline
                    frame[y:y2, x:x2] = patch
                    pert_pred_mask = self.forward(torch.from_numpy(input_copy).to(self.device))[y:y2, x:x2]

                    pert_loss_result = 1 - loss_f.calculate(pert_pred_mask, gt_mask, np.round(baseline, 4))
                    base_loss_result = 1 - loss_f.calculate(base_pred_mask, gt_mask, np.round(baseline, 4))
                    # print(pert_loss_result)
                    # print(base_loss_result)
                    delta = pert_loss_result - base_loss_result
                    # print(delta)
                    importance[ch, y:y2, x:x2][win_mask] = delta / mask_count
        self.normalize_importance(importance)
        self.importance = importance
        return importance

    def get_counter(self, base_input, masks, window):
        b, c, h, w = base_input.shape
        result = [0] * c
        for ch in range(c):
            diff_mask = masks[ch]
            for y in range(0, h, window):
                y2 = min(y + window, h)
                for x in range(0, w, window):
                    x2 = min(x + window, w)
                    win_mask = diff_mask[y:y2, x:x2]
                    if not win_mask.any():
                        continue
                    else:
                        result[ch] += 1
        return result

    def normalize_importance(self, importance):
        c, h, w = importance.shape
        for ch in range(c):
            zero_mask = importance[ch] == 0.0
            min_val = np.min(importance[ch])
            max_val = np.max(importance[ch])
            if max_val > min_val:
                importance[ch] = (importance[ch] - min_val) / (max_val - min_val)
            else:
                importance[ch] = 0.0
            importance[ch][zero_mask] = 0.0



