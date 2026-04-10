from pathlib import Path

from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus, HiResCAM
import torch.nn as nn
import torch
import numpy as np
from gradcam.regression_target import RegressionTarget


class GradCam:

    def __init__(self, model, input_array, module=None, cam_class=GradCAMPlusPlus):
        self.model = model
        self.module_name = module
        self.cam_class = cam_class
        if module is None:
            self.module = self._get_last_layer()
        else:
            self.module = self._find_last(module)
        self.input = input_array
        self.cam_algo = self._get_method()

    def _get_method(self):
        algo = self.cam_class(model=self.model, target_layers=[self.module])
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

    def _find_last(self, selected_module):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name == selected_module:
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

    def _build_isolated_input(self, x_nchw: torch.Tensor, c) -> torch.Tensor:
        device = x_nchw.device
        baseline = torch.log(torch.tensor(0.01, dtype=x_nchw.dtype, device=device))
        x_iso = torch.empty_like(x_nchw, device=device, dtype=x_nchw.dtype)
        x_iso[:] = baseline
        x_iso[:, c, :, :] = x_nchw[:, c, :, :]
        return x_iso

    def _save_cam(self, cam, name: str, title: str):
        cam = np.asarray(cam, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cam, cmap="inferno", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("importance")
        ax.set_title(title)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        plt.tight_layout()
        Path(f"output/gradcam/{self.module_name}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"output/gradcam/{self.module_name}/{name}.png", dpi=150)
        plt.close(fig)

    def _save_cams_grid(self, cams: list):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Isolated Channels Grad-CAM from {self.module_name} layer', fontsize=16)

        axes_flat = axes.flatten()

        for idx, cam in enumerate(cams):
            cam = np.asarray(cam, dtype=np.float32)
            ax = axes_flat[idx]
            im = ax.imshow(cam, cmap="inferno", interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="importance")
            ax.set_title(f'channel {idx + 1}')
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")

        plt.tight_layout()
        Path(f"output/gradcam/{self.module_name}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"output/gradcam/{self.module_name}/cam_grid_2x2.png", dpi=150)
        plt.close(fig)


    def _pred_to_rad(self, pred, from_shape=928, to_shape=900):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        padding = int((from_shape - to_shape) / 2)
        return pred[padding:padding + to_shape, padding:padding + to_shape].copy()

    def run_isolated_channels(self, target: RegressionTarget, aug_smooth: bool = False) -> list:
        self._disable_inplace_relu()
        cams = []
        for c in range(self.input.size(1)):
            print(f"Running for channel {c + 1}!")
            self.cam_algo = self._get_method()
            x_iso = self._build_isolated_input(self.input, c)
            gray = self.cam_algo(input_tensor=x_iso, targets=[target], aug_smooth=aug_smooth, eigen_smooth=False)
            cams.append(gray[0])

        self._save_cams_grid(cams)

        return cams

    def run_all_channels(self, target: RegressionTarget, aug_smooth=False):
        self._disable_inplace_relu()
        self.cam_algo = self._get_method()
        gray = self.cam_algo(input_tensor=self.input, targets=[target], aug_smooth=aug_smooth, eigen_smooth=False)
        self._save_cam(gray[0], "cam_", 'Merged Grad-CAM from all channels')
        return gray[0]
