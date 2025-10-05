from pathlib import Path
import numpy as np
import torch
import utils
from optical_flow.optical_flow import OpticalFlow
from utils import data_preprocessing, data_postprocessing
from rainnet import RainNet

from convert_from_h5 import load_keras_h5_into_torch

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
    """
    Input:
      - (4, H, W)
      - (H, W, 4)
      - (N, 4, H, W)
      - (N, H, W, 4)
    Returns (N, 4, H, W) float32.
    """
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

    scans = [utils.read_ry_radolan(p) for p in file_paths]
    X_raw = np.stack(scans, axis=0).astype("float32")
    print("X_raw min/max:", np.min(X_raw), np.max(X_raw))
    print("Count of NaN in X_raw: ", np.isnan(X_raw).sum())

    for count, image in enumerate(X_raw):
        utils.show_and_save(image, f'input_{count}', f'Input t-{15 - count * 5} (mm/5min)')

    X = data_preprocessing(X_raw)
    assert X.dtype == np.float32
    model = _load_torch_model()
    x_t = _to_torch_input(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    model.to(device)
    x_t = x_t.to(device)
    with torch.inference_mode():
        y_t = model(x_t)
    Y_pred = y_t.squeeze(1).cpu().numpy()
    Y_mm = data_postprocessing(Y_pred)[0]
    utils.show_and_save(Y_mm, "OUT", "Prediction (mm/5min)")
    utils.create_gif()

    scans.append(Y_mm)
    of = OpticalFlow("output/input_0.png", "output/OUT.png", window_size=32, cell=46)
    good0, good1 = of.calculate()
    of.draw("output/OUT.png", good0, good1)


if __name__ == "__main__":
    main()
