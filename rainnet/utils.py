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
boundaries = [0.009, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 7.5, 10.0, 15.0, 23.0, 58.0]
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
    img = np.array(img, dtype=np.float64)
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

    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')
    edited_boundaries = boundaries.copy()
    edited_boundaries[0] = 0.01
    cbar = fig.colorbar(im, ax=ax, ticks=edited_boundaries)
    cbar.set_label("Rain intensity [mm / h]", fontweight="bold")
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("km", fontweight="bold")
    ax.set_ylabel("km", fontweight="bold")
    fig.savefig(f"output/detailed/{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"output/detailed/{name}.svg", format="svg", bbox_inches="tight")
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

    rgba[..., 3] = 0.2
    rgba[mask] = [1.0, 0.0, 0.0, 1.0]

    plt.imsave(f"output/clean/{name}.png", rgba)
    if show:
        plt.figure()
        plt.imshow(rgba, interpolation='nearest')
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

    arr = data.astype("float64", copy=True)

    if sec_idx is not None and np.size(sec_idx) > 0:
        arr.flat[sec_idx] = nodata

    marr = np.ma.masked_equal(arr, nodata)
    arr = np.ma.filled(marr, 0.0).astype("float64")
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


def show_and_save_importance(image, importance, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    rgba = cmap(norm(image))

    importance = np.asarray(importance, dtype=float)

    alpha = np.zeros_like(importance, dtype=float)
    alpha[importance != 0.0] = 1.0

    fig, ax = plt.subplots()
    ax.imshow(rgba, alpha=0.1, interpolation='nearest')

    im = ax.imshow(importance, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation='nearest')
    im.set_alpha(alpha)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Importance", fontsize=9)

    ax.axis("off")
    fig.savefig(f"output/clean/{name}.png", bbox_inches="tight", dpi=100)
    fig.savefig(f"output/clean/{name}.svg", bbox_inches="tight", format="svg")

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_trio(c, img1, img2, img3, name1, name2, name3, thr=0.01, show=False, union_only=True):
    Path(f"output/clean/{c}").mkdir(parents=True, exist_ok=True)

    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    img3 = np.asarray(img3, dtype=float)

    h, w = img3.shape

    def _hitmiss(pred, gt):
        pred_r = pred >= thr
        gt_r = gt >= thr

        region = (pred_r | gt_r) if union_only else np.ones_like(gt_r, dtype=bool)

        correct = (pred_r == gt_r) & region
        wrong = (pred_r != gt_r) & region

        total = int(region.sum())
        ok = int(correct.sum())
        return ok, total, correct, wrong

    def _overlay(ax, correct, wrong):
        ov = np.zeros((h, w, 4), dtype=float)
        ov[correct] = [0.0, 1.0, 0.0, 0.85]
        ov[wrong] = [1.0, 0.0, 0.0, 0.85]
        ax.imshow(ov, interpolation='nearest')

    ok1, tot1, correct1, wrong1 = _hitmiss(img1, img3)
    ok2, tot2, correct2, wrong2 = _hitmiss(img2, img3)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    im0 = ax[0].imshow(img1, cmap=cmap, norm=norm, interpolation='nearest')
    ax[0].set_title(f"{name1}  ({(ok1 / tot1) * 100:.2f} %)")
    ax[0].axis("off")
    _overlay(ax[0], correct1, wrong1)

    ax[1].imshow(img2, cmap=cmap, norm=norm, interpolation='nearest')
    ax[1].set_title(f"{name2}  ({(ok2 / tot2) * 100:.2f} %)")
    ax[1].axis("off")
    _overlay(ax[1], correct2, wrong2)

    ax[2].imshow(img3, cmap=cmap, norm=norm, interpolation='nearest')
    ax[2].set_title(name3)
    ax[2].axis("off")

    cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rain intensity [mm / h]", fontweight="bold")

    out_name = f"{name1}_{name2}_{name3}"
    fig.savefig(f"output/clean/{c}/{out_name}.svg", format="svg", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_cluster(cluster_output, image, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean/cluster").mkdir(parents=True, exist_ok=True)

    label_img, clusters = cluster_output

    image = np.asarray(image, dtype=float)

    base_rgba = cmap(norm(image))
    base_rgba[..., 3] = 0.2

    if label_img.max() == 0:
        return

    n_classes = label_img.max()
    rng = np.random.default_rng(42)
    colors = rng.random((n_classes + 1, 3))

    overlay = np.zeros((*label_img.shape, 4), dtype=float)

    for cls in range(1, n_classes + 1):
        mask = (label_img == cls)
        overlay[mask, :3] = colors[cls]
        overlay[mask, 3] = 1.0

    fig, ax = plt.subplots()
    ax.imshow(base_rgba, interpolation='nearest')
    ax.imshow(overlay, interpolation='nearest')
    ax.axis("off")
    ax.set_aspect("equal")

    fig.savefig(
        f"output/clean/cluster/{name}.svg",
        format="svg",
        bbox_inches="tight"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_cluster_window(cluster, image, x1, y1, x2, y2, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean/cluster").mkdir(parents=True, exist_ok=True)
    label_img, clusters_list = cluster

    image = np.asarray(image, dtype=float)

    base_rgba = cmap(norm(image))
    base_rgba[..., 3] = 0.25

    n_classes = int(label_img.max())
    if n_classes == 0:
        fig, ax = plt.subplots()
        ax.imshow(base_rgba, interpolation='nearest')
        rect = plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                             fill=False, linewidth=2.0, edgecolor="yellow")
        ax.add_patch(rect)
        ax.axis("off")
        fig.savefig(f"output/clean/cluster/{name}.svg", format="svg", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    rng = np.random.default_rng(42)
    colors = rng.random((n_classes + 1, 3))

    overlay = np.zeros((*label_img.shape, 4), dtype=float)
    for cls in range(1, n_classes + 1):
        m = (label_img == cls)
        overlay[m, :3] = colors[cls]
        overlay[m, 3] = 0.95

    fig, ax = plt.subplots()
    ax.imshow(base_rgba, interpolation='nearest')
    ax.imshow(overlay, interpolation='nearest')

    rect = plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                         fill=False, linewidth=2.0, edgecolor="red")
    ax.add_patch(rect)

    ax.axis("off")
    ax.set_aspect("equal")

    fig.savefig(
        f"output/clean/cluster/{name}.svg",
        format="svg",
        bbox_inches="tight"
    )
    if show:
        plt.show()
    else:
        plt.close(fig)
