from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM, HiResCAM
import torch.nn as nn
import torch
import numpy as np

import utils
from gradcam.regression_target import RegressionTarget


class GradCam:

    def __init__(self, model, inputArray,
                 method, module=None):
        self.method = method
        self.model = model
        if module is None:
            self.module = self._get_last_layer()
        else:
            self.module = self._find_last(module)
        self.input = inputArray
        self.cam_algo = self._get_method()

    def _get_method(self):
        if self.method == "gradcam++":
            algo = GradCAMPlusPlus(model=self.model, target_layers=[self.module])
        elif self.method == "hires":
            algo = HiResCAM(model=self.model, target_layers=[self.module])
        elif self.method == "layercam":
            algo = LayerCAM(model=self.model, target_layers=[self.module])
        elif self.method == "eigen":
            algo = EigenCAM(model=self.model, target_layers=[self.module])
        else:
            algo = GradCAM(model=self.model, target_layers=[self.module])
        try:
            algo.relu = False
        except AttributeError:
            pass
        return algo

    def _get_last_layer(self):
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        return None

    def _find_last(self, selectedModule):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name == selectedModule:
                return module
        return None

    def run(self, target: RegressionTarget):
        with self.cam_algo:
            grayscale_cam = self.cam_algo(input_tensor=self.input,
                                          targets=[target],
                                          eigen_smooth=False,
                                          aug_smooth=False)
            return grayscale_cam[0]

    def _disable_inplace_relu(self):
        for m in self.model.modules():
            if isinstance(m, (nn.ReLU, nn.LeakyReLU)) and getattr(m, "inplace", False):
                m.inplace = False

    def _build_isolated_input(self, x_nchw: torch.Tensor, c, bleed) -> torch.Tensor:
        device = x_nchw.device
        baseline = torch.log(torch.tensor(0.01, dtype=x_nchw.dtype, device=device))  # ~ -4.60517
        x_iso = torch.empty_like(x_nchw, device=device, dtype=x_nchw.dtype)
        x_iso[:] = baseline
        if bleed > 0.0:
            x_iso += bleed * (x_nchw - baseline)
        x_iso[:, c, :, :] = x_nchw[:, c, :, :]
        return x_iso

    def _save_cam(self, cam, name: str, title: str):
        cam = np.asarray(cam, dtype=np.float32)
        p99 = np.percentile(cam, 99)
        if not np.isfinite(p99) or p99 <= 0:
            vmin, vmax = float(np.min(cam)), float(np.max(cam))
            if vmax - vmin < 1e-12:
                cam_norm = np.zeros_like(cam, dtype=np.float32)
            else:
                cam_norm = (cam - vmin) / (vmax - vmin)
        else:
            cam_norm = cam / (p99 + 1e-8)
        cam_norm = np.clip(cam_norm, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cam_norm, origin="lower", cmap="inferno",
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("importance (normalized)")
        ax.set_title(title)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        plt.tight_layout()
        fig.savefig(f"output/{name}.png", dpi=150)
        plt.close(fig)

    def _pred_to_rad(self, pred, from_shape=928, to_shape=900):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        padding = int((from_shape - to_shape) / 2)
        return pred[padding:padding + to_shape, padding:padding + to_shape].copy()

    def run_per_channel(self, target: RegressionTarget, bleed=0.05, aug_smooth: bool = False) -> list:
        self._disable_inplace_relu()
        cams = []
        for c in range(self.input.size(1)):
            print(f"Running for channel {c + 1}!")
            self.cam_algo = self._get_method()
            x_iso = self._build_isolated_input(self.input, c, bleed=bleed)
            gray = self.cam_algo(input_tensor=x_iso, targets=[target], aug_smooth=aug_smooth, eigen_smooth=False)
            cams.append(gray[0])
            self._save_cam(gray[0], f"cam_{c + 1}", f'channel{c + 1}')
        return cams

    def _build_single_input(self, x_nchw: torch.Tensor, c, bleed) -> torch.Tensor:
        device = x_nchw.device
        baseline = torch.log(torch.tensor(0.01, dtype=x_nchw.dtype, device=device))
        x_iso = torch.empty_like(x_nchw, device=device, dtype=x_nchw.dtype)
        x_iso[:] = baseline
        if bleed > 0.0:
            x_iso += bleed * (x_nchw - baseline)
        x_iso[:, c, :, :] = x_nchw[:, 0, :, :]  # always use channel 0
        return x_iso

    def test_one_channel(self, target: RegressionTarget, bleed=0.0, aug_smooth: bool = False) -> list:
        self._disable_inplace_relu()
        cams = []
        for c in range(self.input.size(1)):
            print(f"Running for channel 0!")
            self.cam_algo = self._get_method()
            x_iso = self._build_single_input(self.input, c, bleed=bleed)
            gray = self.cam_algo(input_tensor=x_iso, targets=[target], aug_smooth=aug_smooth, eigen_smooth=False)
            cams.append(gray[0])
            self._save_cam(gray[0], f"test_cam_{c + 1}", f'channel{0}')
        return cams

    def overlay_on_input(self, base_img_mm, result_cam, title: str = "Grad-CAM overlay"):
        if result_cam.shape[0] == 928:
            result_cam = self._pred_to_rad(result_cam)

        # podklad: prahovanie a dynamický rozsah
        img = np.where(base_img_mm > 1e-2, base_img_mm, 0.0)
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1.0
        vmax = max(vmax, 0.5)

        # --- normalizácia CAM na 99. percentil ---
        if np.any(result_cam > 0):
            vmin, vhigh = np.percentile(result_cam, [1, 99])
            cam_norm = np.clip((result_cam - vmin) / (vhigh - vmin + 1e-8), 0, 1)
        else:
            cam_norm = result_cam

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.ma.masked_less_equal(img, 0.0),
                  origin="lower",
                  cmap="viridis",
                  vmin=0.0,
                  vmax=vmax,
                  interpolation="nearest")

        # použijeme normalizovaný CAM
        ax.imshow(cam_norm,
                  origin="lower",
                  cmap="Reds",
                  alpha=0.6,
                  interpolation="nearest")

        ax.set_title(title)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        plt.tight_layout()
        fig.savefig(f"output/{title}.png", dpi=150)
        plt.close(fig)
