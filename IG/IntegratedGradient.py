import numpy as np
import meteors as mt
from .scalar_wrapper import ScalarWrapper
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

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

    def calculate_ig_with_noise(self, input, baseline, steps, n_samples, st_devs, method="smoothgrad"):
        # Noise tunnel
        if n_samples <= 0:
            n_samples = 1  # run at least once
        if input.shape[0] != 4:
            x_chw = self.__to_chw(input)
        else:
            x_chw = input
        if method not in ["smoothgrad", "smoothgrad_square", "vargrad"]:
            print("Method not found, switched to default - smoothgrad")
            method = "smoothgrad"
        attrs = []
        for _ in range(n_samples):
            print(f"Running sample number {_ + 1}")
            noisy = x_chw + torch.randn_like(x_chw) * float(st_devs)
            hsi = self.__create_hsi(noisy)
            ig_attr = self.ig.attribute(
                hsi,
                target=0,
                baseline=baseline,
                n_steps=steps,
                return_convergence_delta=False,
                method="gausslegendre",
                internal_batch_size=4
            )
            attrs.append(self.__extract_attr(ig_attr))
        A = torch.stack(attrs, dim=0)
        if method == "smoothgrad":
            A = A.mean(dim=0)
        elif method == "smoothgrad_square":
            A = (A ** 2).mean(dim=0)
        else:
            A = A.var(dim=0, unbiased=False)
        pos, neg = self.__split_values(A)
        pct_pos = self.__channel_percents(pos)
        pct_neg = self.__channel_percents(neg)
        return {
            "attr": A,
            "pct_pos": pct_pos,
            "pct_neg": pct_neg,
            "delta": None
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

    def __threshold_maps(self, pos, neg, value=90.0):
        pos_nonzero = pos[pos > 0]
        neg_nonzero = neg[neg > 0]

        pos_thr = torch.quantile(pos_nonzero, value / 100.0) if pos_nonzero.numel() > 0 else 0.0
        neg_thr = torch.quantile(neg_nonzero, value / 100.0) if neg_nonzero.numel() > 0 else 0.0

        pos = torch.where(pos >= pos_thr, pos, torch.zeros_like(pos))
        neg = torch.where(neg >= neg_thr, neg, torch.zeros_like(neg))
        return pos, neg

    def __global_normalize(self, pos, neg):
        eps = 1e-12
        global_max = torch.max(torch.cat([pos.flatten(), neg.flatten()]))
        pos = pos / (global_max + eps)
        neg = neg / (global_max + eps)
        return pos, neg

    def spatial_heatmaps(self, pos_or_neg, mode="raw"):
        C, H, W = pos_or_neg.shape
        if mode == "raw":
            return pos_or_neg
        elif mode == "channel":
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
        else:
            return None

    def _pred_to_rad(self, pred, from_shape=928, to_shape=900):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        padding = int((from_shape - to_shape) / 2)
        return pred[padding:padding + to_shape, padding:padding + to_shape].copy()

    def show_heatmap(self, images, ig_attr, mode="raw"):
        C, H, W = images.shape
        pos, neg = self.__split_values(ig_attr)
        pos, neg = self.__global_normalize(pos, neg)
        pos, neg = self.__threshold_maps(pos, neg)

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
        pos_raw = [pos_map[i].detach().cpu().numpy() if isinstance(pos_map[i], torch.Tensor) else pos_map[i] for i in
                   range(C)]
        neg_raw = [neg_map[i].detach().cpu().numpy() if isinstance(neg_map[i], torch.Tensor) else neg_map[i] for i in
                   range(C)]
        base_imgs = []

        for i in range(images.shape[0]):
            curr_image = images[i]
            img = np.where(curr_image > 1e-2, curr_image, 0.0)
            base_imgs.append(img)

        def _plot_and_save(kind: str, out_name: str):
            cmap_pos = plt.cm.Blues.copy()
            cmap_neg = plt.cm.Reds.copy()
            cmap_all = plt.cm.RdBu.copy()

            bg = "#e6e6e6"

            cmap_pos.set_bad(bg)
            cmap_neg.set_bad(bg)
            cmap_all.set_bad(bg)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            for ax, i in zip(axes.flat, range(C)):
                ax.imshow(base_imgs[i], cmap="gray")
                ax.set_facecolor(bg)

                p = pos_raw[i]
                n = neg_raw[i]

                if p.shape[0] != 900:
                    p = self._pred_to_rad(p)
                    n = self._pred_to_rad(n)

                if kind == "ALL":
                    signed = p - n
                    signed = np.ma.masked_where(np.abs(signed) < 1e-6, signed)
                    vmax = np.percentile(np.abs(signed.compressed()), 99) if signed.count() > 0 else 1.0
                    im = ax.imshow(signed, cmap=cmap_all, vmin=-vmax, vmax=vmax)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    title = (f"channel {i+1}\n"
                             f"POS:{float(all_pos_share[i]):.2f}% | NEG:{float(all_neg_share[i]):.2f}%")

                elif kind == "POSITIVE":
                    p_masked = np.ma.masked_where(p <= 0, p)
                    vmax = np.percentile(p[p > 0], 99) if np.any(p > 0) else 1.0
                    im = ax.imshow(
                        p_masked,
                        cmap=cmap_pos,
                        norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
                    )
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    title = f"channel {i+1}\nshare of total POS: {float(overall_pos_pct[i]):.2f}%"

                else:
                    n_masked = np.ma.masked_where(n <= 0, n)
                    vmax = np.percentile(n[n > 0], 99) if np.any(n > 0) else 1.0
                    im = ax.imshow(
                        n_masked,
                        cmap=cmap_neg,
                        norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
                    )
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    title = f"channel {i+1}\nshare of total NEG: {float(overall_neg_pct[i]):.2f}%"

                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(kind, fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 1])
            fig.savefig(out_name, dpi=200, bbox_inches="tight")
            plt.close(fig)

        _plot_and_save("ALL", f"../IG/output/rainnet/ig_{mode}_all.png")
        _plot_and_save("POSITIVE", f"../IG/output/rainnet/ig_{mode}_positive.png")
        _plot_and_save("NEGATIVE", f"../IG/output/rainnet/ig_{mode}_negative.png")
