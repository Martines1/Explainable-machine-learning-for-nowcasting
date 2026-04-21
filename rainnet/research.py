import sys
from pathlib import Path
import numpy as np
import torch

from perturbation import difference
from perturbation.cluster import ClusterPerturbation
import utils
from utils import data_preprocessing
from rainnet_arch import RainNet
from convert_from_h5 import load_keras_h5_into_torch

current_file = Path(__file__).resolve()
current_dir = current_file.parent

data_number = "2"

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
    return None


def _to_torch_input(X: np.ndarray) -> torch.Tensor:
    x = None
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

    '''
    0.75 = 0,75 mm/h zrážok v danom pixeli
    '''

    ground_truth = None

    # X_raw shape (C, H, W)
    X_raw = np.stack(scans, axis=0).astype("float32")

    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())

    # data_preprocessing returns (B, H, W, C)
    X = data_preprocessing(X_raw)
    X1 = data_preprocessing(X_raw, False)[0, :, :, :4]
    if X.shape[-1] == 5:
        ground_truth = X[0, :, :, -1]
        X = X[:, :, :, :4]

    model = _load_torch_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert X.dtype == np.float32

    # Convert to (B, C, H, W)
    x_t = _to_torch_input(X)

    model.to(device)
    x_t = x_t.to(device)

    pert = ClusterPerturbation(model, x_t, device, ground_truth)
    threshold = 0.001
    baseline = np.log(0.01)
    rain_thr_log = np.log(0.01 + threshold)
    X1 = np.transpose(X1, (2, 0, 1))
    masks = difference.calculate_diff_unique(X1, 0.0)

    difference.compare_all(X1, 0.0, 2)

    loss_functions = ["logcosh", "MSE", "BMSE", "accuracy"]
    loss_maps = {}
    c = 0
    for loss_function in loss_functions:
        print(f"Processing with loss {loss_function}")

        clusters = []
        for i in range(4):
            clusters.append(pert.cluster_mask_k_means(masks[i], n_clusters=10))

        importance = pert.perturbate_channels(
            baseline=baseline,
            masks=masks,
            thr=rain_thr_log,
            clusters=clusters,
            loss=loss_function,
            weighted=False
        )

        loss_maps[loss_function] = importance[c]

    utils.save_loss_comparison_grid(
        X1[c],
        loss_maps,
        file_name="channel_1_loss_function_comparison",
        title=f"Loss function comparison for channel {c + 1}",
    )


if __name__ == "__main__":
    main()
