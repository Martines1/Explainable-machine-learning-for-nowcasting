import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator


class OpticalFlow:
    def __init__(self, frame1, frame2, window_size, cell):
        self.frame1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
        self.frame2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
        self.window_size = window_size
        self.cell = cell

    def calculate(self):
        h, w = self.frame1.shape

        offset = 20
        size = 10
        ys, xs = np.mgrid[offset:h - offset:size, offset:w - offset:size]

        p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
        p1, st, e = cv2.calcOpticalFlowPyrLK(self.frame1, self.frame2, p0, None,
                                             winSize=(self.window_size, self.window_size),
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
                                             minEigThreshold=0.001, maxLevel=2)
        good1 = p1[st.flatten() == 1]
        good0 = p0[st.flatten() == 1]
        return self.__postprocessing(good0, good1)

    def __postprocessing(self, p0, p1):
        p0, p1 = self.__remove_outliers(p0, p1)
        p0, p1 = self.__downsample_and_center(p0, p1)
        p0, p1 = self.__interpolate(p0, p1)
        p0, p1 = self.__normalize(p0, p1, length=14.0)
        return p0, p1

    def circ_mean(self, a):
        s = np.sin(a).mean()
        c = np.cos(a).mean()
        return np.arctan2(s, c)

    def circ_dist(self, a, b):
        return np.abs(np.arctan2(np.sin(a-b), np.cos(a-b)))

    def __remove_outliers(self, p0, p1):
        p0_flat = p0.reshape(-1, 2)
        p1_flat = p1.reshape(-1, 2)
        p = p1_flat - p0_flat
        lengths = np.linalg.norm(p, axis=1)
        thr = np.percentile(lengths, 95)
        mask = lengths <= thr
        filtered_p0 = p0[mask]
        filtered_p1 = p1[mask]
        return filtered_p0, filtered_p1

    def __downsample_and_center(self, p0, p1):
        p0_flat = p0.reshape(-1, 2).astype(np.float32)
        p1_flat = p1.reshape(-1, 2).astype(np.float32)
        H, W = self.frame1.shape[:2]
        cells_x = int(np.ceil(W / self.cell))
        cells_y = int(np.ceil(H / self.cell))
        d = p1_flat - p0_flat
        angle = np.arctan2(d[:, 1], d[:, 0])
        dx = d[:, 0]
        dy = d[:, 1]
        cells = dict()

        keep = []

        for i in range(len(p0_flat)):
            x, y = p0_flat[i]
            cx = int(x) // self.cell
            cy = int(y) // self.cell
            if cx < 0 or cy < 0 or cx >= cells_x or cy >= cells_y:
                continue
            key = (cx, cy)
            if key not in cells.keys():
                cells[key] = []
            cells[key].append(i)

        for key, idxs in cells.items():
            a = angle[idxs]
            mu = self.circ_mean(a)
            dist = self.circ_dist(a, mu)
            selected_point = idxs[int(np.argmin(dist))]
            keep.append(selected_point)

        for i in keep:
            x, y = p0_flat[i]
            cx = int(x) // self.cell
            cy = int(y) // self.cell
            if 0 <= cx < cells_x and 0 <= cy < cells_y:
                center = (cx * self.cell + self.cell / 2, cy * self.cell + self.cell / 2)
                p0_flat[i] = center
                p1_flat[i] = (center[0] + dx[i], center[1] + dy[i])
        keep = np.array(keep, dtype=np.int32)
        return p0_flat[keep], p1_flat[keep]


    def __interpolate(self, p0, p1):
        p0 = p0.reshape(-1, 2).astype(np.float32)
        p1 = p1.reshape(-1, 2).astype(np.float32)

        H, W = self.frame1.shape[:2]

        xs = np.clip(np.arange(0, W, self.cell) + 0.5 * self.cell, 0, W - 1)
        ys = np.clip(np.arange(0, H, self.cell) + 0.5 * self.cell, 0, H - 1)
        XX, YY = np.meshgrid(xs, ys)
        centers = np.column_stack([XX.ravel(), YY.ravel()])
        disp = (p1 - p0)

        rbf = RBFInterpolator(
            p0,
            disp,
            kernel='gaussian',
            epsilon=4,
            smoothing=0.05,
            neighbors=42
        )
        pred_disp = rbf(centers)
        measured = {(float(x), float(y)): disp[i] for i, (x, y) in enumerate(p0)}
        for j, (x, y) in enumerate(centers):
            t = (float(x), float(y))
            if t in measured:
                pred_disp[j] = measured[t]

        p0_out = centers.astype(np.float32)
        p1_out = (centers + pred_disp).astype(np.float32)
        return p0_out, p1_out

    def __normalize(self, p0, p1, length=14.0):
        p0_flat = p0.reshape(-1, 2).astype(np.float32)
        p1_flat = p1.reshape(-1, 2).astype(np.float32)
        p = p1_flat - p0_flat
        norm = np.linalg.norm(p, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        unit = p / norm
        p2n = (unit * length) + p0_flat
        return p0_flat, p2n

    def draw(self, frame, p0, p1):
        frame = cv2.imread(frame)
        for (pt1, pt2) in zip(p0, p1):
            x1, y1 = pt1.ravel()
            x2, y2 = pt2.ravel()
            cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2, tipLength=0.3)
        cv2.imshow("Lucas-Kanade flow", frame)
        cv2.imwrite("flow.png", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
