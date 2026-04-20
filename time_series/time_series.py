import numpy as np
import matplotlib.pyplot as plt


def vis_time_series(gt_data, last_gt_data=None, predicted_data=None, points=None):
    def _to_thw(arr):
        if arr is None:
            return None

        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 2:
            return arr[None, ...]

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim == 3:
            if arr.shape[-1] <= 10 and arr.shape[0] > 10 and arr.shape[1] > 10:
                arr = np.transpose(arr, (2, 0, 1))
            return arr

        raise ValueError(f"Unsupported shape: {arr.shape}")

    gt_data = _to_thw(gt_data)
    last_gt_data = _to_thw(last_gt_data)
    predicted_data = _to_thw(predicted_data)

    if gt_data is None or gt_data.shape[0] == 0:
        raise ValueError("gt_data must contain at least one frame")

    if points is None:
        first_arr = gt_data[0]
        max_idx = np.argmax(first_arr)
        selected_point = np.unravel_index(max_idx, first_arr.shape)
    else:
        selected_point = points

    y, x = selected_point

    gt_hist = np.log1p(gt_data[:, y, x])

    gt_future = None
    if last_gt_data is not None:
        gt_future = np.log1p(last_gt_data[:, y, x])

    pred_future = None
    if predicted_data is not None:
        pred_future = np.log1p(predicted_data[:, y, x])

    plt.figure(figsize=(7, 4.5))

    x_hist = np.arange(len(gt_hist))
    plt.plot(
        x_hist,
        gt_hist,
        marker='o',
        linestyle='-',
        color='b',
        label='Ground truth history'
    )

    xticks = list(x_hist)

    if gt_future is not None and len(gt_future) > 0:
        x_gt_future = np.arange(len(gt_hist), len(gt_hist) + len(gt_future))
        xticks.extend(list(x_gt_future))

        plt.plot(
            np.concatenate(([len(gt_hist) - 1], x_gt_future)),
            np.concatenate(([gt_hist[-1]], gt_future)),
            marker='o',
            linestyle='-',
            color='g',
            label='Ground truth future'
        )

    if pred_future is not None and len(pred_future) > 0:
        x_pred_future = np.arange(len(gt_hist), len(gt_hist) + len(pred_future))
        xticks.extend(list(x_pred_future))

        plt.plot(
            np.concatenate(([len(gt_hist) - 1], x_pred_future)),
            np.concatenate(([gt_hist[-1]], pred_future)),
            marker='o',
            linestyle='-',
            color='r',
            label='Predicted'
        )

    plt.legend()
    plt.title(
        f"Rain intensity over time for point ({y}, {x})",
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel("time", fontsize=12, fontweight="bold")
    plt.ylabel("log(1 + rain_intensity)", fontsize=12, fontweight="bold")
    plt.xticks(sorted(set(xticks)))
    plt.tight_layout()
    plt.show()