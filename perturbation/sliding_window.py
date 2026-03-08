from perturbation.loss_functions import *
import torch


class SlidingWindowPerturbation:
    def __init__(self, model, input_, device, ground_truth):
        # input_ shape should be (B, C, H, W) or (B, C, W, H)
        self.importance = None
        self.model = model
        self.model.eval()
        self.input = input_
        self.ground_truth = ground_truth
        self.device = device
        self.highest_diff = {"c": {"value": -np.inf, "x": None, "y": None, "x2": None, "y2": None}}
        self.lowest_diff = {"c": {"value": np.inf, "x": None, "y": None, "x2": None, "y2": None}}

    def forward(self, x):
        with torch.inference_mode():
            y_t = self.model(x)
        return y_t.detach().cpu().numpy()[0, 0, :, :]

    def perturbate_channels(self, baseline, masks, thr, loss="logcosh", window=128, stride=32, weighted=False):
        base_input = self.input.detach().cpu().numpy()
        b, c, h, w = base_input.shape

        if stride is None:
            stride = window // 2

        loss_f = get_function(loss)
        base_pred = self.forward(self.input)

        counter = self.get_counter(masks, window, stride, h, w)

        imp_sum = np.zeros((c, h, w), dtype=np.float32)
        imp_cnt = np.zeros((c, h, w), dtype=np.float32)

        print(f"Starting window perturbation for {c} channels")
        for ch in range(c):
            diff_mask = masks[ch]
            internal_counter = 0

            for y in range(0, h, stride):
                y2 = min(y + window, h)
                for x in range(0, w, stride):
                    x2 = min(x + window, w)

                    win_mask = diff_mask[y:y2, x:x2]
                    if not win_mask.any():
                        continue

                    internal_counter += 1
                    print(f'Processing channel {ch}, done {(internal_counter / max(counter[ch], 1)) * 100:.2f} %')

                    gt_mask = self.ground_truth[y:y2, x:x2]
                    base_pred_mask = base_pred[y:y2, x:x2]

                    input_copy = base_input.copy()
                    frame = input_copy[0, ch, :, :]
                    patch = frame[y:y2, x:x2]

                    patch[win_mask] = baseline
                    frame[y:y2, x:x2] = patch

                    pert_pred = self.forward(torch.from_numpy(input_copy).to(self.device))
                    pert_pred_mask = pert_pred[y:y2, x:x2]

                    pert_score = loss_f.calculate(pert_pred_mask, gt_mask, thr)
                    base_score = loss_f.calculate(base_pred_mask, gt_mask, thr)
                    delta = base_score - pert_score
                    # high delta = perturbation made it worse (important),
                    # low delta = perturbation made it better (noise)

                    if np.isnan(pert_score) or np.isnan(base_score):
                        continue
                    if weighted:
                        mask_cnt = int(win_mask.sum())
                        if mask_cnt == 0:
                            continue
                        weighted = delta * mask_cnt
                        self.saveExtreme(ch, weighted, x, y, x2, y2)
                    else:
                        self.saveExtreme(ch, delta, x, y, x2, y2)

                    imp_sum[ch, y:y2, x:x2][win_mask] += delta
                    imp_cnt[ch, y:y2, x:x2][win_mask] += 1.0

        importance = imp_sum / (imp_cnt + 1e-8)
        support = (imp_cnt > 0) & masks
        self.normalize_importance(importance, support)

        self.importance = importance
        return importance

    def get_counter(self, masks, window, stride, h, w):
        c = masks.shape[0]
        result = [0] * c
        for ch in range(c):
            diff_mask = masks[ch]
            for y in range(0, h, stride):
                y2 = min(y + window, h)
                for x in range(0, w, stride):
                    x2 = min(x + window, w)
                    if diff_mask[y:y2, x:x2].any():
                        result[ch] += 1
        return result

    def normalize_importance(self, importance, support, eps=1e-8, q=97, range='local'):
        if range == 'local':
            self.local_normalize_importance(importance, support, eps, q)
        else:
            self.global_normalize_importance(importance, support, eps, q)

    def local_normalize_importance(self, importance, support, eps=1e-8, q=97):
        c, h, w = importance.shape
        for ch in range(c):
            sup = support[ch]
            if not np.any(sup):
                importance[ch].fill(0.0)
                continue

            vals = importance[ch][sup]
            abs_vals = np.abs(vals)

            scale = np.percentile(abs_vals, q)
            if not np.isfinite(scale) or scale < eps:
                importance[ch].fill(0.0)
                continue

            importance[ch][sup] = np.clip(importance[ch][sup] / scale, -1.0, 1.0)
            importance[ch][~sup] = 0.0

    def global_normalize_importance(self, importance, support, eps=1e-8, q=97):
        if support is None or not np.any(support):
            importance.fill(0.0)
            return

        vals = importance[support]
        abs_vals = np.abs(vals)

        scale = np.percentile(abs_vals, q)
        if not np.isfinite(scale) or scale < eps:
            importance.fill(0.0)
            return

        importance[support] = np.clip(importance[support] / scale, -1.0, 1.0)
        importance[~support] = 0.0

    def saveExtreme(self, c, prediction, x, y, x2, y2):
        if c not in self.highest_diff:
            self.highest_diff[c] = {"value": -np.inf, "x": None, "y": None, "x2": None, "y2": None}
        if c not in self.lowest_diff:
            self.lowest_diff[c] = {"value": np.inf, "x": None, "y": None, "x2": None, "y2": None}

        if prediction > self.highest_diff[c]["value"] and prediction > 0:
            self.highest_diff[c] = {"value": prediction, "x": x, "y": y, "x2": x2, "y2": y2}

        if prediction < self.lowest_diff[c]["value"] and prediction < 0:
            self.lowest_diff[c] = {"value": prediction, "x": x, "y": y, "x2": x2, "y2": y2}

    def getLowest(self, c, masks, baseline=np.log(0.01)):
        rec = self.lowest_diff[c]
        if rec["x"] is None:
            return None, None, None
        return self.calculateExtreme(rec, c, masks, baseline)

    def getHighest(self, c, masks, baseline=np.log(0.01)):
        rec = self.highest_diff[c]
        if rec["x"] is None:
            return None, None, None
        return self.calculateExtreme(rec, c, masks, baseline)

    def calculateExtreme(self, rec, c, masks, baseline=np.log(0.01)):
        x, y, x2, y2 = rec["x"], rec["y"], rec["x2"], rec["y2"]
        base_input = self.input.detach().cpu().numpy()
        base_pred = self.forward(self.input)

        win_mask = masks[c][y:y2, x:x2]

        input_copy = base_input.copy()
        input_copy[0, c, y:y2, x:x2][win_mask] = baseline

        pert_pred = self.forward(torch.from_numpy(input_copy).to(self.device))

        return base_pred[y:y2, x:x2], pert_pred[y:y2, x:x2], self.ground_truth[y:y2, x:x2]
