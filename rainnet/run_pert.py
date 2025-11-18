import sys
from pathlib import Path
import numpy as np
import torch

from perturbation import difference
from perturbation.perturbation import Perturbation
import utils
from utils import data_preprocessing, data_postprocessing
from rainnet_arch import RainNet

from convert_from_h5 import load_keras_h5_into_torch

current_file = Path(__file__).resolve()
current_dir = current_file.parent

data_number = "1"

FILES = utils.getData(data_number)
DATA_DIR = current_dir / "data" / data_number

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
    if len(file_paths) < 4:
        sys.exit("RainNet requires 4 radar images as an input!")
    print("Used files as input (ordered by time):")
    for p in file_paths:
        print("  ", p.name)

    scans = [utils.read_ry_radolan(p) for p in file_paths]
    ground_truth = None
    if len(file_paths) == 5:
        utils.show_and_save(scans[4].astype("float32"), f'ground_truth')

    X_raw = np.stack(scans, axis=0).astype("float32")

    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())
    for count, image in enumerate(X_raw):
        utils.show_and_save(image, f'input_{count}')

    # for i in range(3):
    #     mask = difference.calculate_diff(X_raw[i], X_raw[i+1], 0.0)
    #     difference.show_diff(X_raw[i+1], mask)

    # for i in range(4):
    #     mask = difference.calculate_diff_both(X_raw[i], ground_truth, 0.0)
    #     difference.show_diff(X_raw[i], mask)

    X = data_preprocessing(X_raw)
    if X.shape[-1] == 5:
        ground_truth = X[0, :, :, -1]
        X = X[:, :, :, :4]

    model = _load_torch_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X.dtype == np.float32
    x_t = _to_torch_input(X)

    model.to(device)
    x_t = x_t.to(device)
    masks = difference.calculate_diff_unique(X_raw[:4], 0.0)
    for i in range(4):
        difference.show_diff(X_raw[i], masks[i])
    for i in range(1):
        pert = Perturbation(model, x_t, device, ground_truth)
        baseline = data_preprocessing(np.zeros_like(X_raw))[0, :, :, 0]
        print(np.min(baseline), np.max(baseline))
        pert_result = pert.turn_off_channels([1, 2], baseline, "accuracy")
        print(f'Diff {i}: {pert_result["diff"] * 100:.2f} %')
        print(f'Ground truth perputated loss {i}: {pert_result["gt_perp_diff"] * 100:.2f} %')
        print(f'Ground truth base loss {i}: {pert_result["gt_base_diff"] * 100:.2f} %')
    Y_mm = pert_result["base_pred"]
    Y_pert_mm = pert_result["pert_pred"]
    Y_mm = Y_mm[None, :, :]
    Y_pert_mm = Y_pert_mm[None, :, :]
    Y_mm = data_postprocessing(Y_mm, False)[0]
    Y_pert_mm = data_postprocessing(Y_pert_mm, False)[0]
    utils.show_and_save(Y_mm, "OUT_base")
    utils.show_and_save(Y_pert_mm, "OUT_pert")

#


if __name__ == "__main__":
    main()
