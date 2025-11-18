import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from wradlib import io as wio
from pathlib import Path
import os
import re


def parse_ts(fname: str) -> str:
    _TIME_RE = re.compile(r".*-(\d{10})-dwd---bin(?:\.\w+)?$")
    m = _TIME_RE.match(fname)
    if not m:
        print(f"Warning: can't parse time from file name {fname}")
        return fname
    return m.group(1)


def Scaler(array):
    return np.log(array + 0.01)


def invScaler(array):
    return np.exp(array) - 0.01


def pad_to_shape(array, from_shape=900, to_shape=928, how="mirror"):
    padding = int((to_shape - from_shape) / 2)
    array_padded = None
    if how == "zero":
        array_padded = np.pad(array, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant",
                              constant_values=0)
    elif how == "mirror":
        array_padded = np.pad(array, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="reflect")
    return array_padded


def pred_to_rad(pred, from_shape=928, to_shape=900):
    padding = int((from_shape - to_shape) / 2)
    return pred[::, padding:padding + to_shape, padding:padding + to_shape].copy()


def data_preprocessing(X):
    X = np.moveaxis(X, 0, -1)
    X = X[np.newaxis, ::, ::, ::]
    X = Scaler(X)
    X = pad_to_shape(X)

    return X


def data_postprocessing(nwcst, shrink=False):
    if nwcst.ndim == 4:
        nwcst = nwcst[..., 0]
    nwcst = invScaler(nwcst)
    if shrink:
        nwcst = pred_to_rad(nwcst)
    nwcst = np.where(nwcst > 0, nwcst, 0)
    return nwcst


def show_and_save(img, name):
    Path("output").mkdir(parents=True, exist_ok=True)

    img = np.where(img > 1e-2, img, 0.0)
    pos = img[img > 0]
    if pos.size:
        vmax = float(max(np.percentile(pos, 99), 0.5))
    else:
        vmax = 1.0

    norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = cm.get_cmap("viridis")
    rgba = cmap(norm(img))
    mask0 = (img == 0.0)
    rgba[mask0, :3] = 1.0
    rgba[mask0, 3] = 1.0

    plt.imsave(f"output/{name}.png", rgba)


def create_gif():
    files = ["output/input_0.png", "output/input_1.png", "output/input_2.png", "output/input_3.png", "output/OUT.png"]
    frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    frames[0].save(
        "output/out.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
        optimize=True,
        disposal=2
    )


def read_ry_radolan(path: Path) -> np.ndarray:
    data, attrs = wio.read_radolan_composite(str(path), missing=None)
    nodata = attrs.get("nodataflag", -9999)
    sec_idx = attrs.get("secondary")

    arr = data.astype("float32", copy=True)

    if sec_idx is not None and np.size(sec_idx) > 0:
        arr.flat[sec_idx] = nodata

    marr = np.ma.masked_equal(arr, nodata)
    arr = np.ma.filled(marr, 0.0).astype("float32")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape != (900, 900):
        raise ValueError(f"Expected 900x900, got {arr.shape} for {path}")
    return arr


def getData(number):
    data = []
    folder = "data/" + number
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            data.append(filename)
    return data
