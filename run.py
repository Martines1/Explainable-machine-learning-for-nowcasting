import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from wradlib import io as wio
import cartopy.crs as ccrs
# tvoje moduly – ostávajú
from utils import data_preprocessing, data_postprocessing
from pyproj import CRS
import rioxarray

current_file = Path(__file__).resolve()
current_dir = current_file.parent

# --------- KONFIG ---------
FILES = [
    "raa01-ry_10000-2509181830-dwd---bin.bz2",
    "raa01-ry_10000-2509181835-dwd---bin.bz2",
    "raa01-ry_10000-2509181840-dwd---bin.bz2",
    "raa01-ry_10000-2509181845-dwd---bin.bz2",
]
DATA_DIR   = current_dir / "data"
MODEL_PATH = "model/rainnet.h5"
OUT_NPY    = "out_pred_mm.npy"
OUT_PNG    = "out_pred.png"

# RADOLAN názov: ...-YYMMDDHHMM-dwd---bin(.bz2)
_TIME_RE = re.compile(r".*-(\d{10})-dwd---bin(?:\.\w+)?$")

def _parse_ts(fname: str) -> str:
    m = _TIME_RE.match(fname)
    if not m:
        print(f"Varovanie: neviem vyparsovať čas zo súboru: {fname}")
        return fname
    return m.group(1)  # YYMMDDHHMM

def _read_ry_radolan(path: Path) -> np.ndarray:
    """Načíta RADOLAN RY cez wradlib a vráti 2D float32 mm/5min (900×900).
       No-data a 'secondary' pixely sú nastavené na 0 (aby sa dali použiť v modeli)."""
    show_radolan_one(path)
    data, attrs = wio.read_radolan_composite(str(path), missing=None)
    nodata = attrs.get("nodataflag", -9999)
    sec_idx = attrs.get("secondary")

    arr = data.copy()
    if sec_idx is not None and np.size(sec_idx) > 0:
        arr.flat[sec_idx] = nodata

    # maskuj no-data a vyplň nulou (model ju znesie; pri zobrazení nulu maskujeme)
    marr = np.ma.masked_equal(arr, nodata)
    arr = np.ma.filled(marr, 0.0).astype("float32")

    if arr.shape != (900, 900):
        raise ValueError(f"Očakávané 900×900, dostal som {arr.shape} pre {path}")
    return arr

def show_radolan_one(path: Path):
    RADOLAN_CRS = CRS.from_proj4(
        "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +k=1 "
        "+x_0=0 +y_0=0 +a=6370040 +b=6370040 +units=m +no_defs"
    )
    ds = wio.open_radolan_dataset(str(path))
    da = ds[list(ds.data_vars)[0]].squeeze().astype("float32")   # typicky "RY"

    # pozadie (bez zrážok) skryjeme – nech je transparentné
    da = da.where(da > 0)

    # nastav správne CRS + scalar nodata a reprojektuj
    da = da.rio.write_crs(RADOLAN_CRS)
    da = da.rio.write_nodata(np.nan)  # <- kľúčové: scalar nodata
    da_ll = da.rio.reproject("EPSG:4326", nodata=np.nan, resampling=0)  # nearest

    # vykreslenie na mapu
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
    gl.right_labels = False; gl.top_labels = False
    ax.set_title(path.name)
    plt.tight_layout(); plt.show()

def _save_png(path, arr_mm):
    try:
        from PIL import Image
    except Exception:
        print("Upozornenie: Pillow nie je nainštalované – PNG neuložím.")
        return
    v = np.clip(arr_mm, 0, 10)
    v = (v / 10.0 * 255.0).astype(np.uint8)
    Image.fromarray(v).save(path)

def main():
    # 0) zoradenie podľa timestampu
    file_paths = [DATA_DIR / f for f in FILES]
    file_paths = sorted(file_paths, key=lambda p: _parse_ts(p.name))

    print("Použité vstupy (zoradené časovo):")
    for p in file_paths:
        print("  ", p.name)

    # 1) načítanie 4 po sebe idúcich 5-min kompozitov -> (4,900,900)
    scans = [_read_ry_radolan(p) for p in file_paths]
    X_raw = np.stack(scans, axis=0).astype("float32")  # (4,900,900)

    # 3) predspracovanie pre RainNet (log/padding/kanály-last/batch)
    X = data_preprocessing(X_raw)  # očak.: (1,928,928,4)

    # 4) inferencia
    model = load_model(MODEL_PATH, compile=False)
    Y_pred = model.predict(X, verbose=1)

    # 5) spätné škálovanie a orez na 900×900 (mm/5min)
    Y_mm = data_postprocessing(Y_pred)  # (900,900)

    # 6) uloženie
    np.save(OUT_NPY, Y_mm)
    print(f"Uložené: {OUT_NPY}  (numpy, mm/5min, predikcia t+5)")

    if OUT_PNG:
        _save_png(OUT_PNG, Y_mm)
        if os.path.exists(OUT_PNG):
            print(f"Uložené: {OUT_PNG}  (náhľad PNG, 0–10 mm)")

if __name__ == "__main__":
    main()
