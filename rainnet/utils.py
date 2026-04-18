import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator
from wradlib import io as wio
from pathlib import Path
import os
import re
from matplotlib.patches import Patch


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


def show_and_save(img, file_name, title, show=False):
    img = np.array(img, dtype=np.float64)
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    Path("output/detailed").mkdir(parents=True, exist_ok=True)

    plt.imsave(f"output/clean/{file_name}.png", cmap(norm(img)))

    h, w = img.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        img,
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper',
        extent=(0, w, 0, h)
    )

    edited_boundaries = boundaries.copy()
    edited_boundaries[0] = 0.01
    cbar = fig.colorbar(im, ax=ax, ticks=edited_boundaries)
    cbar.set_label("Rain intensity [mm / h]", fontsize=15, fontweight="bold")

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("X (pixels)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y (pixels)", fontsize=13, fontweight="bold")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks(np.arange(0, w + 1, 200))
    ax.set_yticks(np.arange(0, h + 1, 200))
    ax.tick_params(axis='both', labelsize=13)

    fig.savefig(f"output/detailed/{file_name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"output/detailed/{file_name}.svg", format="svg", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


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


def show_and_save_importance(image, importance, name, title, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)

    if "window" in name.split("_"):
        Path("output/clean/window_perturbation").mkdir(parents=True, exist_ok=True)
        path_method = "window_perturbation"
    elif "kmeans" in name.split("_"):
        Path("output/clean/kmeans").mkdir(parents=True, exist_ok=True)
        path_method = "kmeans"
    else:
        Path("output/clean/dbscan").mkdir(parents=True, exist_ok=True)
        path_method = "dbscan"

    rgba = cmap(norm(image))
    importance = np.asarray(importance, dtype=float)
    h, w = importance.shape

    alpha = np.zeros_like(importance, dtype=float)
    alpha[importance != 0.0] = 1.0

    fig, ax = plt.subplots()

    ax.imshow(
        rgba,
        alpha=0.1,
        interpolation='nearest',
        origin='upper',
        extent=(0, w, 0, h)
    )

    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xlabel("X (pixels)", fontsize=16)
    ax.set_ylabel("Y (pixels)", fontsize=16)
    ax.tick_params(labelsize=14)

    im = ax.imshow(
        importance,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest',
        origin='upper',
        extent=(0, w, 0, h)
    )
    im.set_alpha(alpha)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks(np.arange(0, w + 1, 200))
    ax.set_yticks(np.arange(0, h + 1, 200))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Importance", fontsize=16)

    fig.savefig(f"output/clean/{path_method}/{name}.png", bbox_inches="tight", dpi=100)
    fig.savefig(f"output/clean/{path_method}/{name}.svg", bbox_inches="tight", format="svg")

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_trio(c, img1, img2, img3, name1, name2, name3, method, thr=0.01, union_only=True):
    Path("output/perturbation").mkdir(parents=True, exist_ok=True)
    Path(f"output/perturbation/{method}").mkdir(parents=True, exist_ok=True)
    Path(f"output/perturbation/{method}/{c}").mkdir(parents=True, exist_ok=True)

    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    img3 = np.asarray(img3, dtype=float)

    h, w = img3.shape

    def _evaluate(pred_r, gt_r, region):
        correct = pred_r & gt_r & region
        wrong = (pred_r ^ gt_r) & region
        total = int(correct.sum() + wrong.sum())
        ok = int(correct.sum())
        return ok, total, correct, wrong

    def _status(pred_r, gt_r, region):
        s = np.full(gt_r.shape, -1, dtype=np.int8)
        s[(pred_r ^ gt_r) & region] = 0
        s[(~pred_r & ~gt_r) & region] = 1
        s[(pred_r & gt_r) & region] = 2
        return s

    def _change_quality(base_r, pert_r, gt_r, region):
        base_status = _status(base_r, gt_r, region)
        pert_status = _status(pert_r, gt_r, region)

        changed = (base_r != pert_r) & region
        changed_to_worse = changed & (pert_status < base_status)
        changed_to_better = changed & (pert_status > base_status)

        return changed_to_worse, changed_to_better

    def _overlay(ax, correct, wrong, changed_bad=None, changed_good=None):
        ov = np.zeros((h, w, 4), dtype=float)
        ov[correct] = [0.0, 1.0, 0.0, 0.85]
        ov[wrong] = [1.0, 0.0, 0.0, 0.85]

        if changed_bad is not None:
            ov[changed_bad] = [1.0, 0.6, 0.0, 0.95]

        if changed_good is not None:
            ov[changed_good] = [0.0, 0.0, 1.0, 0.95]

        ax.imshow(ov, interpolation='nearest')

    def _style_axis(ax, x1, x2, y1, y2):
        ax.set_xlim(x1 - 0.5, x2 + 0.5)
        ax.set_ylim(y2 + 0.5, y1 - 0.5)

        ax.set_xlabel("X (pixels)", fontsize=13)
        ax.set_ylabel("Y (pixels)", fontsize=13)

        ax.set_xticks(np.arange(x1 - 0.5, x2 + 1.5, 1), minor=True)
        ax.set_yticks(np.arange(y1 - 0.5, y2 + 1.5, 1), minor=True)

        ax.grid(False)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.2, alpha=0.35)

        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

    base_r = img1 >= thr
    pert_r = img2 >= thr
    gt_r = img3 >= thr

    if union_only:
        region = base_r | pert_r | gt_r
    else:
        region = np.ones_like(gt_r, dtype=bool)

    ok1, tot1, correct1, wrong1 = _evaluate(base_r, gt_r, region)
    ok2, tot2, correct2, wrong2 = _evaluate(pert_r, gt_r, region)

    changed_bad, changed_good = _change_quality(base_r, pert_r, gt_r, region)
    x1, x2, y1, y2 = __find_best_area(img1, img2, img3, thr=thr, padding=5)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle(f"Channel {c + 1} difference", fontsize=18, fontweight="bold")

    im0 = ax[0].imshow(img1, cmap=cmap, norm=norm, interpolation='nearest')
    ax[0].set_facecolor("lightgray")
    ax[0].set_title(f"{name1} ({(ok1 / tot1) * 100:.2f} %)" if tot1 > 0 else f"{name1} (0.00 %)", fontsize=16, fontweight="bold")
    _overlay(ax[0], correct1, wrong1)
    _style_axis(ax[0], x1, x2, y1, y2)

    ax[1].imshow(img2, cmap=cmap, norm=norm, interpolation='nearest')
    ax[1].set_facecolor("lightgray")
    ax[1].set_title(f"{name2} ({(ok2 / tot2) * 100:.2f} %)" if tot2 > 0 else f"{name2} (0.00 %)", fontsize=16, fontweight="bold")
    _overlay(ax[1], correct2, wrong2, changed_bad, changed_good)
    _style_axis(ax[1], x1, x2, y1, y2)

    ax[2].imshow(img3, cmap=cmap, norm=norm, interpolation='nearest')
    ax[2].set_title(name3, fontsize=16, fontweight="bold")
    _style_axis(ax[2], x1, x2, y1, y2)

    legend_handles = [
        Patch(facecolor="red", edgecolor="red", label="Wrong"),
        Patch(facecolor="green", edgecolor="green", label="Correct"),
        Patch(facecolor="orange", edgecolor="orange", label=f"Changed to worse ({int(changed_bad.sum())})"),
        Patch(facecolor="blue", edgecolor="blue", label=f"Changed to better ({int(changed_good.sum())})"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=4,
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.15, -0.06)
    )

    cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rain intensity [mm / h]", fontweight="bold")

    out_name = f"{name1}_{name2}_{name3}"
    fig.savefig(
        f"output/perturbation/{method}/{c}/{out_name}.svg",
        format="svg",
        bbox_inches="tight"
    )
    plt.close(fig)


def __find_best_area(img1, img2, img3, thr=0.01, padding=5):
    def _bbox(img):
        mask = np.asarray(img, dtype=float) >= thr
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            h, w = img.shape
            return 0, w - 1, 0, h - 1

        x1 = max(0, xs.min() - padding)
        x2 = min(img.shape[1] - 1, xs.max() + padding)
        y1 = max(0, ys.min() - padding)
        y2 = min(img.shape[0] - 1, ys.max() + padding)
        return x1, x2, y1, y2

    b1 = _bbox(img1)
    b2 = _bbox(img2)
    b3 = _bbox(img3)

    x1 = min(b1[0], b2[0], b3[0])
    x2 = max(b1[1], b2[1], b3[1])
    y1 = min(b1[2], b2[2], b3[2])
    y2 = max(b1[3], b2[3], b3[3])

    return x1, x2, y1, y2


def save_cluster(cluster_output, image, name, title, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/perturbation").mkdir(parents=True, exist_ok=True)
    Path("output/perturbation/cluster").mkdir(parents=True, exist_ok=True)

    label_img, clusters = cluster_output

    image = np.asarray(image, dtype=float)
    h, w = label_img.shape

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

    ax.imshow(
        base_rgba,
        interpolation='nearest',
        origin='upper',
        extent=(0, w, 0, h)
    )

    im = ax.imshow(
        overlay,
        interpolation='nearest',
        origin='upper',
        extent=(0, w, 0, h)
    )

    ax.set_xlabel("X (pixels)", fontweight="bold")
    ax.set_ylabel("Y (pixels)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks(np.arange(0, w + 1, 200))
    ax.set_yticks(np.arange(0, h + 1, 200))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([])
    cbar.set_label("Cluster ID", fontweight="bold")

    fig.savefig(
        f"output/perturbation/cluster/{name}.svg",
        format="svg",
        bbox_inches="tight"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_cluster_window(cluster, image, x1, y1, x2, y2, name, show=False):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path(f"output/perturbation").mkdir(parents=True, exist_ok=True)
    Path("output/perturbation/cluster").mkdir(parents=True, exist_ok=True)

    label_img, clusters_list = cluster
    image = np.asarray(image, dtype=float)

    h, w = label_img.shape

    crop_w = w // 2
    crop_h = h // 2

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left = int(round(cx - crop_w / 2))
    right = left + crop_w
    top = int(round(cy - crop_h / 2))
    bottom = top + crop_h

    if left < 0:
        left = 0
        right = crop_w
    if right > w:
        right = w
        left = w - crop_w
    if top < 0:
        top = 0
        bottom = crop_h
    if bottom > h:
        bottom = h
        top = h - crop_h

    left = max(0, left)
    right = min(w, right)
    top = max(0, top)
    bottom = min(h, bottom)

    base_rgba = cmap(norm(image))
    base_rgba[..., 3] = 0.25

    n_classes = int(label_img.max())

    fig, ax = plt.subplots()

    if n_classes == 0:
        ax.imshow(base_rgba, interpolation='nearest')

        rect = plt.Rectangle(
            (x1, y1), (x2 - x1), (y2 - y1),
            fill=False, linewidth=2.0, edgecolor="yellow"
        )
        ax.add_patch(rect)

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.axis("off")
        ax.set_aspect("equal")

        fig.savefig(
            f"output/perturbation/cluster/{name}.svg",
            format="svg",
            bbox_inches="tight"
        )

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

    ax.imshow(base_rgba, interpolation='nearest')
    ax.imshow(overlay, interpolation='nearest')

    rect = plt.Rectangle(
        (x1, y1), (x2 - x1), (y2 - y1),
        fill=False, linewidth=2.0, edgecolor="red"
    )
    ax.add_patch(rect)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    ax.set_aspect("equal")

    fig.savefig(
        f"output/perturbation/cluster/{name}.svg",
        format="svg",
        bbox_inches="tight"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_importance_grid(data, pert_result, file_name, title):
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/perturbation").mkdir(parents=True, exist_ok=True)

    if "window" in file_name.split("_"):
        Path("output/perturbation/window_perturbation").mkdir(parents=True, exist_ok=True)
        path_method = "window_perturbation"
    elif "kmeans" in file_name.split("_"):
        Path("output/perturbation/kmeans").mkdir(parents=True, exist_ok=True)
        path_method = "kmeans"
    else:
        Path("output/perturbation/dbscan").mkdir(parents=True, exist_ok=True)
        path_method = "dbscan"

    frames = np.asarray(data, dtype=float)
    importances = np.asarray(pert_result, dtype=float)

    if frames.ndim == 4 and frames.shape[0] == 1:
        frames = frames[0]
    if importances.ndim == 4 and importances.shape[0] == 1:
        importances = importances[0]
    if frames.ndim == 3 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if importances.ndim == 3 and importances.shape[-1] == 1:
        importances = importances[..., 0]

    C, h, w = frames.shape

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes_flat = axes.flatten()

    last_im = None

    for i in range(4):
        ax = axes_flat[i]

        if i < C:
            rgba = cmap(norm(frames[i]))
            importance = np.asarray(importances[i], dtype=float)

            alpha = np.zeros_like(importance, dtype=float)
            alpha[importance != 0.0] = 1.0

            vis_importance = np.sign(importance) * (np.abs(importance) ** 0.7)

            ax.imshow(
                rgba,
                alpha=0.09,
                interpolation='nearest',
                origin='upper',
                extent=[0, w, 0, h]
            )

            im = ax.imshow(
                vis_importance,
                cmap="RdYlBu_r",
                vmin=-1.0,
                vmax=1.0,
                interpolation='nearest',
                origin='upper',
                extent=[0, w, 0, h]
            )
            im.set_alpha(alpha)
            last_im = im

            ax.set_title(f"Channel {i + 1}", fontweight="bold", fontsize=16)
            ax.set_xlabel("X (pixels)", fontsize=16)
            ax.set_ylabel("Y (pixels)", fontsize=16)
            ax.tick_params(labelsize=14)
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_xticks(np.arange(0, w + 1, 200))
            ax.set_yticks(np.arange(0, h + 1, 200))
        else:
            ax.axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=18)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.08)
    cbar.set_label("Importance", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.savefig(f"output/perturbation/{path_method}/{file_name}.png", bbox_inches="tight", dpi=100)
    fig.savefig(f"output/perturbation/{path_method}/{file_name}.svg", bbox_inches="tight", format="svg")
    plt.close(fig)
