import sys
from datetime import datetime
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
    if len(file_paths) == 5:
        current_file = file_paths[4]
        ts = utils.parse_ts(current_file.name)

        dt = datetime.strptime(ts, "%y%m%d%H%M")
        pretty = dt.strftime("%d.%m.%Y %H:%M")

        utils.show_and_save(scans[4].astype("float32"), f'Ground truth', title=f'Ground truth {pretty}')
    # X_raw shape (C, H, W)
    X_raw = np.stack(scans, axis=0).astype("float32")

    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())
    for count, image in enumerate(X_raw[:4]):
        current_file = file_paths[count]
        ts = utils.parse_ts(current_file.name)

        dt = datetime.strptime(ts, "%y%m%d%H%M")
        pretty = dt.strftime("%d.%m.%Y %H:%M")
        utils.show_and_save(image, f'Input_{count}', f'Input #{count + 1} - {pretty}')

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

    clusters = []
    method = "kmeans"
    for i in range(4):
        # DBSCAN:
        #  clusters.append(pert.cluster_mask_dbscan(masks[i], eps=20.0, min_cluster_size=10))

        # KMeans:
        clusters.append(pert.cluster_mask_k_means(masks[i], n_clusters=2))

        utils.save_cluster(clusters[i], X1[i], f'cluster_{i}', f'DBSCAN clustering of channel {i+1}')

        x1, y1, x2, y2 = pert.create_window(clusters[i][1][0], padding=16)
        utils.show_cluster_window(clusters[i], X1[i], x1, y1, x2, y2, f'cluster_window_{i}')

    importance = pert.perturbate_channels(
        baseline=baseline,
        masks=masks,
        thr=rain_thr_log,
        clusters=clusters,
        loss="accuracy",
        weighted=False
    )

    for i in range(4):
        lowest_base, lowest_pert, lowest_gt = pert.get_lowest(i, masks, baseline=baseline)
        if lowest_base is None or lowest_pert is None or lowest_gt is None:
            print(f"Channel {i}: No valid lowest perturbation found.")
        else:
            lowest_base = utils.invScaler(lowest_base)
            lowest_pert = utils.invScaler(lowest_pert)
            lowest_gt = utils.invScaler(lowest_gt)
            utils.show_trio(i, lowest_base, lowest_pert, lowest_gt, "Lowest base", "Lowest perturbation", "Ground truth"
                            , method, thr=threshold)

        highest_base, highest_pert, highest_gt = pert.get_highest(i, masks, baseline=baseline)
        if highest_base is None or highest_pert is None or highest_gt is None:
            print(f"Channel {i}: No valid highest perturbation found.")
        else:
            highest_base = utils.invScaler(highest_base)
            highest_pert = utils.invScaler(highest_pert)
            highest_gt = utils.invScaler(highest_gt)
            utils.show_trio(i, highest_base, highest_pert, highest_gt, "Highest base", "Highest perturbation",
                            "Ground truth", method, thr=threshold)
        if method == "dbscan":
            utils.show_and_save_importance(X1[i], importance[i], f"dbscan_importance_map_{i}",
                                           f"Perturbation of channel {i+1} using DBSCAN clustering", False)
        else:
            utils.show_and_save_importance(X1[i], importance[i], f"kmeans_importance_map_{i}",
                                           f"Perturbation of channel {i+1} using K-means clustering", False)
    if method == "dbscan":
        utils.save_importance_grid(X1, importance, "dbscan_importance_map_grid",
                                   "Perturbation of channels using DBSCAN clustering")
    else:
        utils.save_importance_grid(X1, importance, "kmeans_importance_map_grid",
                                   "Perturbation of channels using K-means clustering")
    np.save('importance_map_cluster.npy', importance)


if __name__ == "__main__":
    main()
