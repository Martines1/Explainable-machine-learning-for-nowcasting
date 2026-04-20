import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as f
import math
import time
import time_series.time_series
import utils
from optical_flow.optical_flow import OpticalFlow
from utils import data_preprocessing, data_postprocessing
from rainnet_arch import RainNet
from perturbation.loss_functions import LogCosh, MSE, BMSE, RainAccuracy
from convert_from_h5 import load_keras_h5_into_torch

current_file = Path(__file__).resolve()
current_dir = current_file.parent

data_number = "2"

FILES = utils.getData(data_number)
DATA_DIR = current_dir / "data" / data_number

PT_WEIGHTS = current_dir / "model" / "rainnet_torch_converted.pt"
H5_WEIGHTS = current_dir / "model" / "rainnet.h5"


def logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = pred - target
    return (x + f.softplus(-2.0 * x) - math.log(2.0)).mean()


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

    current_file = file_paths[3]
    ts = utils.parse_ts(current_file.name)
    end_dt = datetime.strptime(ts, "%y%m%d%H%M")
    end_dt_plus5 = end_dt + timedelta(minutes=5)
    end_dt_plus5 = end_dt_plus5.strftime("%d.%m.%Y %H:%M")

    time_whole = time.time()
    X = data_preprocessing(X_raw[:4])
    assert X.dtype == np.float32
    model = _load_torch_model()
    x_t = _to_torch_input(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    model.to(device)
    x_t = x_t.to(device)
    start_pure_time = time.time()
    with torch.inference_mode():
        y_t = model(x_t)
    elapsed_time = time.time() - start_pure_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\nExecution time: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s total)")

    Y_pred = y_t.squeeze(1).cpu().numpy()
    Y_mm = data_postprocessing(Y_pred, False)[0]

    elapsed_whole_time = time.time() - time_whole
    whole_minutes = int(elapsed_whole_time // 60)
    whole_seconds = elapsed_whole_time % 60

    print(f"\nExecution time with processing: {whole_minutes}m {whole_seconds:.2f}s ("
          f"{elapsed_whole_time:.2f}s total)")

    if len(file_paths) == 5:
        gt_raw = scans[4].astype("float32")
        gt_proc = data_preprocessing(gt_raw[None, ...], scale=True)

        gt_chw = np.transpose(gt_proc[0], (2, 0, 1))
        gt_t = torch.from_numpy(gt_chw[None, ...]).to(device)

        with torch.inference_mode():
            loss_val = logcosh_loss(y_t, gt_t)
        print("RainNet logcosh loss:", float(loss_val))

        pred_np = y_t.squeeze(1).cpu().numpy()[0]
        gt_np = gt_t.squeeze(1).cpu().numpy()[0]
        gt_mm = data_postprocessing(gt_np[None, ...], False)[0]

        mse_val = MSE().calculate(pred_np, gt_np, no_rain=None)
        bmse_val = BMSE().calculate(pred_np, gt_np, no_rain=0.01)
        rain_acc_val = RainAccuracy().calculate(Y_mm, gt_mm, no_rain=0.01)
        logcsosh_val = LogCosh().calculate(pred_np, gt_np)
        print("RainNet MSE:", mse_val)
        print("RainNet balanced MSE:", bmse_val)
        print("RainNet rain accuracy:", rain_acc_val)
        print("LogCosh:", logcsosh_val)
    #  Y_mm = data_postprocessing(Y_pred, True)[0]
    #  scans.append(Y_mm)

    utils.show_and_save(Y_mm, "out", title=f"Predicted precipitation {end_dt_plus5}")
    # utils.create_gif()

    inputs_ts = data_preprocessing(X_raw[:4], scale=False)[0]
    inputs_ts = np.transpose(inputs_ts, (2, 0, 1)).astype("float32")

    last_gt_ts = None
    if len(file_paths) == 5:
        gt_ts = data_preprocessing(scans[4][None, ...], scale=False)[0]
        last_gt_ts = np.transpose(gt_ts, (2, 0, 1)).astype("float32")

    predicted_ts = Y_mm[None, ...].astype("float32")

    time_series.time_series.vis_time_series(
        gt_data=inputs_ts,
        last_gt_data=last_gt_ts,
        predicted_data=predicted_ts
    )

    # Optical Flow part
    # of = OpticalFlow("output/clean/input_0.png", "output/clean/input_3.png", window_size=32, cell=46)
    # good0, good1 = of.calculate()
    # of.draw("output/clean/input_3.png", good0, good1)

    # Print execution time


if __name__ == "__main__":
    main()
