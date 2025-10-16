import numpy as np
import meteors as mt
from .scalar_wrapper import ScalarWrapper
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class IntegratedGradient:

    def __init__(self, model, target):
        self.model = model
        self.target = target
        self.model.eval()
        self.scalar_model = ScalarWrapper(self.model, self.target).eval()
        self.explainable = mt.models.ExplainableModel(self.scalar_model, "regression")
        self.ig = mt.attr.IntegratedGradients(self.explainable)

    def calculate_ig(self, input, baseline, steps):
        if input.shape[0] != 4:
            x_chw = self.__to_chw(input)
        else:
            x_chw = input
        hsi = self.__create_hsi(x_chw)
        ig_attr = self.ig.attribute(
            hsi,
            target=0,
            baseline=baseline,
            n_steps=steps,
            return_convergence_delta=True,
            method="gausslegendre",
            internal_batch_size=4
        )
        A = self.__extract_attr(ig_attr)
        pos, neg = self.__split_values(A)
        pct_pos = self.__channel_percents(pos)
        pct_neg = self.__channel_percents(neg)
        return {
            "attr": A,
            "pct_pos": pct_pos,
            "pct_neg": pct_neg,
            "delta": getattr(ig_attr, "score", None)
        }

    def __to_chw(self, image):
        # from (H, W, C) to (C, H, W)
        arr = np.moveaxis(image, -1, 0)
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __create_hsi(self, image):
        wavelength = [15, 10, 5, 0]  # in minutes
        hsi = mt.HSI(image=image, wavelengths=wavelength, orientation="CHW")
        return hsi

    def __extract_attr(self, attr_obj):
        if isinstance(attr_obj, torch.Tensor):
            return attr_obj
        for key in ("attributes", "attr", "values"):
            if hasattr(attr_obj, key):
                val = getattr(attr_obj, key)
                return val if torch.is_tensor(val) else torch.as_tensor(val)

    def __split_values(self, ig_attr):
        pos = torch.clamp(ig_attr, min=0)
        neg = -torch.clamp(ig_attr, max=0)
        return pos, neg

    def __channel_percents(self, value):
        ch_sum = value.view(value.size(0), -1).sum(dim=1)
        total = ch_sum.sum()
        if total <= 0:
            return torch.zeros_like(ch_sum)
        return 100.0 * ch_sum / total

    def spatial_heatmaps(self, pos_or_neg, mode="channel"):
        C, H, W = pos_or_neg.shape
        if mode == "channel":
            maps = []
            for c in range(C):
                m = pos_or_neg[c]
                s = m.sum()
                maps.append(m / s if s > 0 else torch.zeros_like(m))
            return torch.stack(maps, dim=0)
        elif mode == "global":
            total = pos_or_neg.sum()
            if total > 0:
                return pos_or_neg / total
            else:
                return torch.zeros_like(pos_or_neg)

    def __norm_heatmap(self, result_cam):
        if isinstance(result_cam, torch.Tensor):
            result_cam = result_cam.detach().cpu().numpy()
        if np.any(result_cam > 0):
            vmin, vhigh = np.percentile(result_cam, [1, 99])
            cam_norm = np.clip((result_cam - vmin) / (vhigh - vmin + 1e-8), 0, 1)
        else:
            cam_norm = result_cam
        return cam_norm

    def _pred_to_rad(self, pred, from_shape=928, to_shape=900):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        padding = int((from_shape - to_shape) / 2)
        return pred[padding:padding + to_shape, padding:padding + to_shape].copy()

    def show_heatmap(self, images, ig_attr, channels=["t-15", "t-10", "t-5", "t-0"], mode="channel"):
        C, H, W = images.shape
        pos, neg = self.__split_values(ig_attr)

        pos_sum_per_channel = pos.sum(dim=(1, 2))
        neg_sum_per_channel = neg.sum(dim=(1, 2))
        eps = 1e-12
        all_pos_share = 100.0 * pos_sum_per_channel / (pos_sum_per_channel + neg_sum_per_channel + eps)
        all_neg_share = 100.0 - all_pos_share

        sum_pos_all = pos_sum_per_channel.sum() + eps
        overall_pos_pct = 100.0 * pos_sum_per_channel / sum_pos_all

        sum_neg_all = neg_sum_per_channel.sum() + eps
        overall_neg_pct = 100.0 * neg_sum_per_channel / sum_neg_all

        pos_map = self.spatial_heatmaps(pos, mode=mode)
        neg_map = self.spatial_heatmaps(neg, mode=mode)
        pos_norm = [self.__norm_heatmap(pos_map[i]) for i in range(C)]
        neg_norm = [self.__norm_heatmap(neg_map[i]) for i in range(C)]
        base_imgs, vmaxs = [], []
        for i in range(images.shape[0]):
            curr_image = images[i]
            img = np.where(curr_image > 1e-2, curr_image, 0.0)
            vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1.0
            vmax = max(vmax, 0.5)
            base_imgs.append(img)
            vmaxs.append(vmax)

        def _plot_and_save(kind: str, out_name: str):
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for ax, i in zip(axes.flat, range(C)):
                ax.imshow(base_imgs[i], cmap="gray", origin="lower", vmin=0.0, vmax=vmaxs[i])
                if pos_norm[i].shape[0] != 900:
                    pos_norm[i] = self._pred_to_rad(pos_norm[i])
                    neg_norm[i] = self._pred_to_rad(neg_norm[i])
                    title = (f"{channels[i]}\n"
                             f"POS:{float(all_pos_share[i]):.2f}%  |  "
                             f"NEG:{float(all_neg_share[i]):.2f}%")
                if kind == "ALL":
                    ax.imshow(pos_norm[i], cmap="Reds", origin="lower", alpha=0.55, vmin=0.0, vmax=1.0)
                    ax.imshow(neg_norm[i], cmap="Blues", origin="lower", alpha=0.55, vmin=0.0, vmax=1.0)
                elif kind == "POSITIVE":
                    ax.imshow(pos_norm[i], cmap="Reds", origin="lower", alpha=0.80, vmin=0.0, vmax=1.0)
                    title = (f"{channels[i]}\n"
                             f"share of total POS: {float(overall_pos_pct[i]):.2f}%")
                else:
                    ax.imshow(neg_norm[i], cmap="Blues", origin="lower", alpha=0.80, vmin=0.0, vmax=1.0)
                    title = (f"{channels[i]}\n"
                             f"share of total NEG: {float(overall_neg_pct[i]):.2f}%")
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{kind}", fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 1])
            fig.savefig(out_name, dpi=200, bbox_inches="tight")
            plt.close(fig)
        _plot_and_save("ALL", f"../IG/output/rainnet/ig_{mode}_all.png")
        _plot_and_save("POSITIVE", f"../IG/output/rainnet/ig_{mode}_positive.png")
        _plot_and_save("NEGATIVE",  f"../IG/output/rainnet/ig_{mode}_negative.png")
