import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

def calculate(frame1, frame2, window_size=32):
    frame1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
    #  p0 = cv2.goodFeaturesToTrack(frame1, 10000, 0.01, 10, useHarrisDetector=False)
    h, w = frame1.shape
    ys, xs = np.mgrid[4:h - 4:10, 4:w - 4:10]
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    p1, st, e = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, winSize=(window_size, window_size),
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                                         minEigThreshold=0.001, maxLevel=2)
    good1 = p1[st.flatten() == 1]
    good0 = p0[st.flatten() == 1]
    good0, good1 = remove_outliers(good0, good1)
    return good0, good1


def remove_outliers(p0, p1):
    p0_flat = p0.reshape(-1, 2)
    p1_flat = p1.reshape(-1, 2)
    p = p1_flat - p0_flat
    lengths = np.linalg.norm(p, axis=1)
    thr = np.percentile(lengths, 90)
    mask = lengths <= thr
    filtered_p0 = p0[mask]
    filtered_p1 = p1[mask]
    return filtered_p0, filtered_p1


def grid_points_centered(p0, p1, frame, cell=24, fixed_len=None, eps=1e-3, clip=True):
    H, W = frame.shape[:2]

    p0f = np.asarray(p0, dtype=np.float32).reshape(-1, 2)
    p1f = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    if p0f.size == 0:
        return p0[:0], p1[:0]

    x = np.clip(p0f[:, 0], 0, W - 1)
    y = np.clip(p0f[:, 1], 0, H - 1)
    gx = (x // cell).astype(np.int32)
    gy = (y // cell).astype(np.int32)

    nx = (W + cell - 1) // cell
    keys = gy * nx + gx

    order = np.random.permutation(len(p0f))
    seen = set()
    keep = []
    for idx in order:
        k = keys[idx]
        if k not in seen:
            seen.add(k)
            keep.append(idx)
    idx_keep = np.asarray(keep, dtype=np.int64)

    p0k = p0f[idx_keep]
    p1k = p1f[idx_keep]
    gxx = gx[idx_keep]
    gyy = gy[idx_keep]
    left = gxx * cell
    top = gyy * cell
    cw = np.minimum(cell, W - left)
    ch = np.minimum(cell, H - top)
    cx = left + cw / 2.0
    cy = top + ch / 2.0
    centers = np.stack([cx, cy], axis=1).astype(np.float32)
    d = p1k - p0k
    if fixed_len is not None:
        n = np.linalg.norm(d, axis=1, keepdims=True)
        u = np.zeros_like(d, dtype=np.float32)
        mask = (n[:, 0] > eps)
        u[mask] = d[mask] / n[mask]
        d = u * float(fixed_len)

    p0_out = centers
    p1_out = centers + d

    if clip:
        p1_out[:, 0] = np.clip(p1_out[:, 0], 0, W - 1)
        p1_out[:, 1] = np.clip(p1_out[:, 1], 0, H - 1)

    if np.asarray(p0).ndim == 3:
        p0_out = p0_out.reshape(-1, 1, 2)
        p1_out = p1_out.reshape(-1, 1, 2)

    return p0_out, p1_out

def rbf_fill_missing_cells(good0, good1, frame, cell=24, arrow_len=14,
                           kernel="thin_plate_spline", neighbors=25, smoothing=0.0,
                           eps=1e-6, clip=True):
    H, W = frame.shape[:2]

    P0 = np.asarray(good0, dtype=np.float32).reshape(-1, 2)
    P1 = np.asarray(good1, dtype=np.float32).reshape(-1, 2)
    if P0.size == 0:
        return good0[:0], good1[:0]

    D = P1 - P0

    nx = (W + cell - 1) // cell
    ny = (H + cell - 1) // cell
    lefts = np.arange(nx) * cell
    tops  = np.arange(ny) * cell
    cws = np.minimum(cell, W - lefts)
    chs = np.minimum(cell, H - tops)
    cx = lefts + cws / 2.0
    cy = tops  + chs / 2.0
    CX, CY = np.meshgrid(cx, cy)
    centers_all = np.stack([CX.ravel(), CY.ravel()], axis=1).astype(np.float32)

    nb = None if neighbors is None else max(1, min(neighbors, len(P0)))
    rbf_u = RBFInterpolator(P0, D[:, 0], kernel=kernel, neighbors=nb, smoothing=smoothing)
    rbf_v = RBFInterpolator(P0, D[:, 1], kernel=kernel, neighbors=nb, smoothing=smoothing)
    U = rbf_u(centers_all)
    V = rbf_v(centers_all)
    DV = np.stack([U, V], axis=1).astype(np.float32)

    n = np.linalg.norm(DV, axis=1, keepdims=True)
    n[n < eps] = 1.0
    Uhat = DV / n

    P0_all = centers_all
    P1_all = centers_all + Uhat * float(arrow_len)

    if clip:
        P1_all[:, 0] = np.clip(P1_all[:, 0], 0, W - 1)
        P1_all[:, 1] = np.clip(P1_all[:, 1], 0, H - 1)
    if np.asarray(good0).ndim == 3:
        P0_all = P0_all.reshape(-1, 1, 2)
        P1_all = P1_all.reshape(-1, 1, 2)

    return P0_all, P1_all

def draw(frame_path, good0, good1):
    frame = cv2.imread(frame_path)
    good0, good1 = grid_points_centered(good0, good1, frame, cell=42, fixed_len=15, eps=0.5)
    good0, good1 = rbf_fill_missing_cells(good0, good1, frame, cell=42, arrow_len=15,
                                          kernel="thin_plate_spline", neighbors=25, smoothing=0.0)
    for p0, p1 in zip(good0, good1):
        x0, y0 = p0.ravel()
        x1, y1 = p1.ravel()
        cv2.arrowedLine(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2, tipLength=0.3)

    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame
