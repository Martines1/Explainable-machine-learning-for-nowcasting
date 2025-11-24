import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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


def data_preprocessing(X, scale=True):
    X = np.moveaxis(X, 0, -1)
    X = X[np.newaxis, ::, ::, ::]
    if scale:
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


# 0.005 instead of 0.01 due to numerical instability
boundaries = [0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 7.5, 10.0, 15.0, 23.0, 58.0]
colors = [
    (0.56, 0.71, 1),  # 0.01 – 0.05
    (0.329, 0.553, 1),  # 0.05 – 1
    (0.192, 0.463, 1),  # 0.05 - 0.1
    (0.102, 0.384, 0.949),  # 0.1 - 0.2
    (0.463, 1, 0.463),  # 0.2 - 0.3
    (0.231, 0.788, 0.298),  # 0.3 - 0.5
    "green",  # 1.1 - 5
    "yellow",  # 5.0 - 7.5
    (1, 0.757, 0),  # 7.5 - 10
    (1, 0.549, 0),  # 15 – 23
    "red",  # 23 – 58
    "purple"  # > 58
]

cmap = ListedColormap(colors)
cmap.set_under("white")
norm = BoundaryNorm(boundaries, ncolors=cmap.N)


def show_and_save(img, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    Path("output/detailed").mkdir(parents=True, exist_ok=True)
    # CLEAN VERSION
    plt.imsave(f"output/clean/{name}.png", cmap(norm(img)))

    # DETAILED VERSION
    h, w = img.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(img, cmap=cmap, norm=norm)
    edited_boundaries = boundaries.copy()
    edited_boundaries[0] = 0.01
    cbar = fig.colorbar(im, ax=ax, ticks=edited_boundaries)
    cbar.set_label("Rain intensity [mm / h]", fontweight="bold")
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("km", fontweight="bold")
    ax.set_ylabel("km", fontweight="bold")
    fig.savefig(f"output/detailed/{name}.png", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def show_and_save_mask(img, mask, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    rgba = cmap(norm(img))

    # sanity check
    white = np.array([1.0, 1.0, 1.0, 1.0])
    problem_pixels = np.logical_and(
        mask,
        np.all(rgba == white, axis=-1)
    )
    count = np.sum(problem_pixels)
    if count > 0:
        print(f"Warning: {count} problem pixels in mask visualization for {name}")

    rgba[..., 3] = 0.4
    rgba[mask] = [1.0, 0.0, 0.0, 1.0]

    plt.imsave(f"output/clean/{name}.png", rgba)
    if show:
        plt.figure()
        plt.imshow(rgba)
        plt.axis("off")
        plt.title(name)
        plt.show()
    else:
        plt.close()


def create_gif():
    types = ["clean", "detailed"]
    for t in types:
        files = [f"output/{t}/input #0.png", f"output/{t}/input #1.png", f"output/{t}/input #2.png",
                 f"output/{t}/input #3.png", f"output/{t}/out.png"]
        frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        frames[0].save(
            f"output/{t}/out.gif",
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
