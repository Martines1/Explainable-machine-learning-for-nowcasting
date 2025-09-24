import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import re
from pathlib import Path
import numpy as np
import utils
from rainnet import rainnet
from utils import data_preprocessing, data_postprocessing

current_file = Path(__file__).resolve()
current_dir = current_file.parent

FILES = utils.getData()
DATA_DIR = current_dir / "data"
MODEL_PATH = "model/rainnet.h5"
OUT_NPY = "out_pred_mm.npy"
OUT_PNG = "out_pred.png"

_TIME_RE = re.compile(r".*-(\d{10})-dwd---bin(?:\.\w+)?$")


def _parse_ts(fname: str) -> str:
    m = _TIME_RE.match(fname)
    if not m:
        print(f"Warning: can't parse time from file name {fname}")
        return fname
    return m.group(1)


def main():
    file_paths = [DATA_DIR / f for f in FILES]
    file_paths = sorted(file_paths, key=lambda f: _parse_ts(f.name))

    print("Used files as input (ordered by time):")
    for p in file_paths:
        print("  ", p.name)
    scans = [utils.read_ry_radolan(p) for p in file_paths]
    X_raw = np.stack(scans, axis=0).astype("float32")
    for count, image in enumerate(X_raw):
        utils.show_and_save(image, f'input_{count}', f'Input t-{15 - count * 5} (mm/5min)')
    X = data_preprocessing(X_raw)
    assert X.dtype == np.float32
    model = rainnet()
    model.load_weights("model/rainnet_weights.h5")
    Y_pred = model.predict(X)
    Y_mm = data_postprocessing(Y_pred)[0]
    utils.show_and_save(Y_mm, "OUT", "t+5 RainNet (mm/5min)")
    utils.create_gif()


if __name__ == "__main__":
    main()
