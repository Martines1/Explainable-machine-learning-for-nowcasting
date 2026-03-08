import sys
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt

from perturbation import difference
from perturbation.perturbation import Perturbation
import utils
from utils import data_preprocessing, data_postprocessing
from rainnet_arch import RainNet
import wradlib as wrl
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

    '''
    0.75 = 0,75 mm/h zrážok v danom pixeli
    '''

    ground_truth = None
    if len(file_paths) == 5:
        utils.show_and_save(scans[4].astype("float32"), f'Ground truth')
    # X_raw shape (C, H, W)
    X_raw = np.stack(scans, axis=0).astype("float32")

    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())
    for count, image in enumerate(X_raw[:4]):
        utils.show_and_save(image, f'Input #{count}')

    # for i in range(3):
    #     mask = difference.calculate_diff(X_raw[i], X_raw[i+1], 0.0)
    #     difference.show_diff(X_raw[i+1], mask)

    # for i in range(4):
    #     mask = difference.calculate_diff_both(X_raw[i], ground_truth, 0.0)
    #     difference.show_diff(X_raw[i], mask)

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

    # for i in range(3):
    #     pert = Perturbation(model, x_t, device, ground_truth)
    #     baseline = data_preprocessing(np.zeros_like(X_raw))[0, :, :, 0]
    #
    #     pert_result = pert.turn_off_channels([i, i+1], baseline, "accuracy", np.log(0.02))
    #     print(f'Ground truth base loss: {pert_result["gt_base_diff"] * 100:.2f} %')
    #     print(f'Ground truth perputated loss {i}: {pert_result["gt_perp_diff"] * 100:.2f} %')
    #     print()

    # Y_pert_mm = pert_result["pert_pred"]

    # Y_pert_mm = Y_pert_mm[None, :, :]

    # Y_pert_mm = data_postprocessing(Y_pert_mm, False)[0]
    # utils.show_and_save(Y_pert_mm, "OUT_pert")
    #

    pert = Perturbation(model, x_t, device, ground_truth)
    threshold = 0.001
    baseline = np.log(0.01)
    rain_thr_log = np.log(0.01 + threshold)
    X1 = np.transpose(X1, (2, 0, 1))
    masks = difference.calculate_diff_unique(X1, 0.0)
    for i in range(4):
        utils.show_and_save(X1[i], f'Preprocessed input #{i}')

    ## RUN WINDOW
    # pert_result = pert.perturbate_channels_window(baseline, masks, rain_thr_log, "accuracy", weighted=True)
    # for i in range(1):
        # lowest_base, lowest_pert, lowest_gt = pert.getLowest(i, masks)
        # lowest_base = utils.invScaler(lowest_base)
        # lowest_pert = utils.invScaler(lowest_pert)
        # lowest_gt = utils.invScaler(lowest_gt)
        #
        # highest_base, highest_pert, highest_gt = pert.getHighest(i, masks)
        # highest_base = utils.invScaler(highest_base)
        # highest_pert = utils.invScaler(highest_pert)
        # highest_gt = utils.invScaler(highest_gt)
        #
        # utils.show_trio(i, lowest_base, lowest_pert, lowest_gt, "lowest_base", "lowest_pert", "lowest_gt", thr=threshold)
        # utils.show_trio(i, highest_base, highest_pert, highest_gt, "highest_base", "highest_pert", "highest_gt", thr=threshold)
        # utils.show_and_save_importance(X1[i], pert_result[i], f"importance_map_{i}", False)
    # np.save('importance_map.npy', pert_result)

    ## RUN CLUSTER
    clusters = []
    for i in range(4):
        # DBSCAN:
        # clusters.append(pert.clusterMaskDBSCAN(masks[i], eps=7.0, min_cluster_size=10))

        # KMeans:
        clusters.append(pert.clusterMaskKMeans(masks[i], n_clusters=3))

        utils.save_cluster(clusters[i], X1[i], f'cluster_{i}')

        x1, y1, x2, y2 = pert.createWindow(clusters[i][1][0], padding=16)
        utils.show_cluster_window(clusters[i], X1[i], x1, y1, x2, y2, f'cluster_window_{i}')
    importance = pert.perturbate_channels_cluster(
        baseline=baseline,
        masks=masks,
        thr=rain_thr_log,
        clusters=clusters,
        loss="accuracy",
        weighted=False
    )
    for i in range(4):
        lowest_base, lowest_pert, lowest_gt = pert.getLowest(i, masks)
        if lowest_base is None or lowest_pert is None or lowest_gt is None:
            print(f"Channel {i}: No valid perturbation found.")
        else:
            lowest_base = utils.invScaler(lowest_base)
            lowest_pert = utils.invScaler(lowest_pert)
            lowest_gt = utils.invScaler(lowest_gt)
            utils.show_trio(i, lowest_base, lowest_pert, lowest_gt, "lowest_base", "lowest_pert", "lowest_gt",
                            thr=threshold)

        highest_base, highest_pert, highest_gt = pert.getHighest(i, masks)
        if highest_base is None or highest_pert is None or highest_gt is None:
            print(f"Channel {i}: No valid perturbation found.")
        else:
            highest_base = utils.invScaler(highest_base)
            highest_pert = utils.invScaler(highest_pert)
            highest_gt = utils.invScaler(highest_gt)
            utils.show_trio(i, highest_base, highest_pert, highest_gt, "highest_base", "highest_pert", "highest_gt",
                            thr=threshold)

        utils.show_and_save_importance(X1[i], importance[i], f"importance_map_{i}", False)


if __name__ == "__main__":
    main()
