from perturbation.loss_functions import *
import torch
from sklearn.cluster import DBSCAN, KMeans


class ClusterPerturbation:
    def __init__(self, model, input_, device, ground_truth):
        # input_ shape should be (B, C, H, W) or (B, C, W, H)
        self.importance = None
        self.model = model
        self.model.eval()
        self.input = input_
        self.ground_truth = ground_truth
        self.device = device
        self.highest_diff = {"c": {"value": -np.inf, "x": None, "y": None, "x2": None, "y2": None}}
        self.lowest_diff = {"c": {"value": np.inf, "x": None, "y": None, "x2": None, "y2": None}}

    def forward(self, x):
        with torch.inference_mode():
            y_t = self.model(x)
        return y_t.detach().cpu().numpy()[0, 0, :, :]

    def perturbate_channels(self, baseline, masks, thr, clusters, loss="logcosh", weighted=False):
        base_input = self.input.detach().cpu().numpy()
        b, c, h, w = base_input.shape

        loss_f = get_function(loss)
        base_pred = self.forward(self.input)

        print(f"Starting cluster perturbation for {c} channels")

        imp_sum = np.zeros((c, h, w), dtype=np.float32)
        imp_cnt = np.zeros((c, h, w), dtype=np.float32)

        for ch in range(c):
            label_img, ch_clusters = clusters[ch]

            total = len(ch_clusters)
            internal_counter = 0

            for pts in ch_clusters:
                internal_counter += 1
                print(f"Processing channel {ch}, done {(internal_counter / max(total, 1)) * 100:.2f} %")

                x1, y1, x2, y2 = self.create_window(pts, padding=32)

                gt_mask = self.ground_truth[y1:y2, x1:x2]
                base_pred_mask = base_pred[y1:y2, x1:x2]

                input_copy = base_input.copy()
                ys = pts[:, 0].astype(int)
                xs = pts[:, 1].astype(int)
                input_copy[0, ch, ys, xs] = baseline

                pert_pred = self.forward(torch.from_numpy(input_copy).to(self.device))
                pert_pred_mask = pert_pred[y1:y2, x1:x2]

                pert_score = loss_f.calculate(pert_pred_mask, gt_mask, thr)
                base_score = loss_f.calculate(base_pred_mask, gt_mask, thr)
                delta = base_score - pert_score
                if weighted:
                    wdelta = delta * int(len(pts))
                    self.save_extreme(ch, wdelta, x1, y1, x2, y2, pts)
                else:
                    self.save_extreme(ch, delta, x1, y1, x2, y2, pts)

                imp_sum[ch, ys, xs] += delta
                imp_cnt[ch, ys, xs] += 1.0
        importance = imp_sum / (imp_cnt + 1e-8)
        support = (imp_cnt > 0) & masks
        self.normalize_importance(importance, support, scope='global')

        self.importance = importance
        return importance

    def get_counter(self, masks, window, stride, h, w):
        c = masks.shape[0]
        result = [0] * c
        for ch in range(c):
            diff_mask = masks[ch]
            for y in range(0, h, stride):
                y2 = min(y + window, h)
                for x in range(0, w, stride):
                    x2 = min(x + window, w)
                    if diff_mask[y:y2, x:x2].any():
                        result[ch] += 1
        return result

    def normalize_importance(self, importance, support, eps=1e-8, q=97, scope='local'):
        if scope == 'local':
            self.local_normalize_importance(importance, support, eps, q)
        else:
            self.global_normalize_importance(importance, support, eps, q)

    def local_normalize_importance(self, importance, support, eps=1e-8, q=97):
        c, h, w = importance.shape
        for ch in range(c):
            sup = support[ch]
            if not np.any(sup):
                importance[ch].fill(0.0)
                continue

            vals = importance[ch][sup]
            abs_vals = np.abs(vals)

            scale = np.percentile(abs_vals, q)
            if not np.isfinite(scale) or scale < eps:
                importance[ch].fill(0.0)
                continue

            importance[ch][sup] = np.clip(importance[ch][sup] / scale, -1.0, 1.0)
            importance[ch][~sup] = 0.0

    def global_normalize_importance(self, importance, support, eps=1e-8, q=97):
        if support is None or not np.any(support):
            importance.fill(0.0)
            return

        vals = importance[support]
        abs_vals = np.abs(vals)

        scale = np.percentile(abs_vals, q)
        if not np.isfinite(scale) or scale < eps:
            importance.fill(0.0)
            return

        importance[support] = np.clip(importance[support] / scale, -1.0, 1.0)
        importance[~support] = 0.0

    def save_extreme(self, c, prediction, x, y, x2, y2, pts):
        if c not in self.highest_diff:
            self.highest_diff[c] = {"value": -np.inf, "x": None, "y": None, "x2": None, "y2": None, "pts": None}
        if c not in self.lowest_diff:
            self.lowest_diff[c] = {"value": np.inf, "x": None, "y": None, "x2": None, "y2": None, "pts": None}

        if prediction > self.highest_diff[c]["value"] and prediction > 0:
            self.highest_diff[c] = {"value": prediction, "x": x, "y": y, "x2": x2, "y2": y2, "pts": pts}

        if prediction < self.lowest_diff[c]["value"] and prediction < 0:
            self.lowest_diff[c] = {"value": prediction, "x": x, "y": y, "x2": x2, "y2": y2, "pts": pts}

    def get_lowest(self, c, masks, baseline=np.log(0.01)):
        rec = self.lowest_diff[c]
        if rec["x"] is None:
            return None, None, None
        return self.calculate_extreme(rec, c, masks, baseline)

    def get_highest(self, c, masks, baseline=np.log(0.01)):
        rec = self.highest_diff[c]
        if rec["x"] is None:
            return None, None, None
        return self.calculate_extreme(rec, c, masks, baseline)

    def calculate_extreme(self, rec, c, masks, baseline=np.log(0.01)):
        x, y, x2, y2 = rec["x"], rec["y"], rec["x2"], rec["y2"]
        pts = rec["pts"]

        base_input = self.input.detach().cpu().numpy()
        base_pred = self.forward(self.input)

        input_copy = base_input.copy()
        ys = pts[:, 0].astype(int)
        xs = pts[:, 1].astype(int)
        input_copy[0, c, ys, xs] = baseline

        pert_pred = self.forward(torch.from_numpy(input_copy).to(self.device))

        return base_pred[y:y2, x:x2], pert_pred[y:y2, x:x2], self.ground_truth[y:y2, x:x2]

    def create_window(self, cluster, padding=8):
        if cluster is None or len(cluster) == 0:
            return None, None, None, None

        base_input = self.input.detach().cpu().numpy()
        _, _, h, w = base_input.shape

        ys = cluster[:, 0].astype(int)
        xs = cluster[:, 1].astype(int)

        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())

        y1 = max(y_min - int(padding), 0)
        y2 = min(y_max + int(padding) + 1, h)
        x1 = max(x_min - int(padding), 0)
        x2 = min(x_max + int(padding) + 1, w)

        return x1, y1, x2, y2

    def cluster_mask_dbscan(self, mask, eps=2.0, min_cluster_size=10):
        mask = (mask.astype(bool))

        coords = np.argwhere(mask)
        if coords.size == 0:
            label_img = np.zeros(mask.shape, dtype=np.int32)
            return label_img, []

        db = DBSCAN(
            eps=float(eps),
            min_samples=int(min_cluster_size),
            metric="euclidean"
        )
        labels = db.fit_predict(coords)
        unique = [lab for lab in np.unique(labels) if lab != -1]

        label_img = np.zeros(mask.shape, dtype=np.int32)
        clusters = []
        for class_id, lab in enumerate(unique, start=1):
            pts = coords[labels == lab]  # (Mi, 2)
            if pts.shape[0] < int(min_cluster_size):
                continue

            label_img[pts[:, 0], pts[:, 1]] = class_id
            clusters.append(pts)

        return label_img, clusters

    def cluster_mask_k_means(self, mask, n_clusters=5):
        mask = mask.astype(bool)

        coords = np.argwhere(mask)
        if coords.size == 0:
            label_img = np.zeros(mask.shape, dtype=np.int32)
            return label_img, []

        N = coords.shape[0]
        k = int(n_clusters)

        k = max(1, min(k, N))

        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(coords)

        label_img = np.zeros(mask.shape, dtype=np.int32)
        clusters = []

        for class_id in range(1, k + 1):
            pts = coords[labels == (class_id - 1)]
            if pts.size == 0:
                continue
            label_img[pts[:, 0], pts[:, 1]] = class_id
            clusters.append(pts)

        return label_img, clusters
