from pathlib import Path
import numpy as np
import torch
import utils
from IG.ROI.ROI import ROI
from IG.ROI.ROI_IG import RegressionTargetROI
from utils import data_preprocessing, data_postprocessing
from rainnet import RainNet
from convert_from_h5 import load_keras_h5_into_torch
from IG.IntegratedGradient import IntegratedGradient
from IG.Regression_target import RegressionTargetIG

current_file = Path(__file__).resolve()
current_dir = current_file.parent

FILES = utils.getData()
DATA_DIR = current_dir / "data"

PT_WEIGHTS = current_dir / "model" / "rainnet_torch_converted.pt"
H5_WEIGHTS = current_dir / "model" / "rainnet.h5"


def _load_torch_model():
    if PT_WEIGHTS.exists():
        m = RainNet(in_channels=4)
        sd = torch.load(PT_WEIGHTS, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m
    elif H5_WEIGHTS.exists():
        m = load_keras_h5_into_torch(str(H5_WEIGHTS), in_channels=4)
        m.eval()
        return m


def _to_torch_input(X: np.ndarray) -> torch.Tensor:
    if X.ndim == 3:
        if X.shape[0] == 4:
            x = X[None, ...]
        elif X.shape[-1] == 4:
            x = np.transpose(X, (2, 0, 1))[None, ...]

    elif X.ndim == 4:
        if X.shape[1] == 4:
            x = X
        elif X.shape[-1] == 4:
            x = np.transpose(X, (0, 3, 1, 2))

    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x).contiguous()


def main():
    file_paths = [DATA_DIR / f for f in FILES]
    file_paths = sorted(file_paths, key=lambda f: utils.parse_ts(f.name))

    print("Used files as input (ordered by time):")
    for p in file_paths:
        print("  ", p.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    scans = [utils.read_ry_radolan(p) for p in file_paths]
    X_raw = np.stack(scans, axis=0).astype("float32")
    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())

    for count, image in enumerate(X_raw):
        utils.show_and_save(image, f'input_{count}')

    X = data_preprocessing(X_raw)[0]
    assert X.dtype == np.float32
    target = RegressionTargetIG(mode="mean")

    # roi = ROI("output/OUT.png", radius=8)
    # roi.show_image()
    # mask = roi.get_mask()
    # target = RegressionTargetROI(torch.from_numpy(mask), mode="mean")

    baseline = data_preprocessing(np.zeros_like(X_raw))[0]
    baseline_chw = np.moveaxis(baseline, -1, 0).astype(np.float32)

    # perm = [3, 0, 1, 2]
    #
    # X_perm = X[..., perm]
    # B_perm = baseline_chw[perm, ...]
    # print(X_perm.shape)
    # print(X.shape)
    # print(baseline_chw.shape)
    # print(B_perm.shape)
    baseline_t = torch.from_numpy(baseline_chw).to(device)
    model = _load_torch_model()

    ig_explainer = IntegratedGradient(model, target)
    available_methods = ["smoothgrad", "smoothgrad_square", "vargrad"]
    res = ig_explainer.calculate_ig_with_noise(X, baseline=baseline_t, steps=2, n_samples=4, st_devs=0.01, method=available_methods[1])
    # res = ig_explainer.calculate_ig(X, baseline_t, steps=2)
    X_chw = np.transpose(X, (2, 0, 1)).astype(np.float32, copy=False)
    x_t_chw = torch.from_numpy(X_chw).to(device)

    A = res["attr"]
    print("Convergence:", res["delta"])
    ig_explainer.show_heatmap(X_raw, A, mode="channel")
    x_t = _to_torch_input(X)

    model.to(device)
    x_t = x_t.to(device)
    with torch.inference_mode():
        y_t = model(x_t)
    Y_pred = y_t.squeeze(1).cpu().numpy()
    Y_mm = data_postprocessing(Y_pred)[0]


if __name__ == "__main__":
    main()
