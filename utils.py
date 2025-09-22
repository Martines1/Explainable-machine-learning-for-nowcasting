import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wradlib import io as wio
import cartopy.crs as ccrs
from pyproj import CRS
from pathlib import Path
import os


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


def data_postprocessing(nwcst):
    if nwcst.ndim == 4:
        nwcst = nwcst[..., 0]
    nwcst = invScaler(nwcst)
    nwcst = pred_to_rad(nwcst)
    nwcst = np.where(nwcst > 0, nwcst, 0)
    return nwcst


def show_and_save(img, name, title):
    img = np.where(img > 1e-2, img, 0.0)
    pos = img[img > 0]
    if pos.size:
        vmax = float(np.percentile(pos, 99))
        vmax = max(vmax, 0.5)
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        np.ma.masked_less_equal(img, 0.0),
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest"
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("mm / 5 min")
    ax.set_title(title)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    plt.tight_layout()

    fig.savefig(f"{name}.png", dpi=150)
    plt.show()


def create_gif():
    files = ["input_0.png", "input_1.png", "input_2.png", "input_3.png", "OUT.png"]
    frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    frames[0].save(
        "out.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
        optimize=True,
        disposal=2
    )


def read_ry_radolan(path: Path) -> np.ndarray:
    # show_radolan_one(path)
    data, attrs = wio.read_radolan_composite(str(path), missing=None)
    nodata = attrs.get("nodataflag", -9999)
    sec_idx = attrs.get("secondary")

    arr = data.copy()
    if sec_idx is not None and np.size(sec_idx) > 0:
        arr.flat[sec_idx] = nodata
    marr = np.ma.masked_equal(arr, nodata)
    arr = np.ma.filled(marr, 0.0).astype("float32")

    if arr.shape != (900, 900):
        raise ValueError(f"Expected 900x900, got {arr.shape} for {path}")
    return arr


def show_radolan_one(path: Path):
    RADOLAN_CRS = CRS.from_proj4(
        "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +k=1 "
        "+x_0=0 +y_0=0 +a=6370040 +b=6370040 +units=m +no_defs"
    )
    ds = wio.open_radolan_dataset(str(path))
    da = ds[list(ds.data_vars)[0]].squeeze().astype("float32")
    da = da.where(da > 0)
    da = da.rio.write_crs(RADOLAN_CRS)
    da = da.rio.write_nodata(np.nan)
    da_ll = da.rio.reproject("EPSG:4326", nodata=np.nan, resampling=0)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = da_ll.plot.imshow(
        ax=ax, transform=ccrs.PlateCarree(),
        robust=True, cbar_kwargs={"label": "mm / 5 min"}
    )
    ax.coastlines(resolution="10m", linewidth=0.8)
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6)
    gl.right_labels = False
    gl.top_labels = False
    ax.set_title(path.name)
    plt.tight_layout()
    plt.show()


def getData():
    data = []
    folder = "data"
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            data.append(filename)
    return data
