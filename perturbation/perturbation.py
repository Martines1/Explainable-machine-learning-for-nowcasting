import sys

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
            return None
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
