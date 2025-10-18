import numpy as np
import matplotlib.pyplot as plt

def vis_time_series(data, points=None):
    if data.shape[0] == 1:
        data = data[0]
    if data.shape[-1] == 5:
        data = np.transpose(data, (2, 0, 1))
    if points is None:
        first_arr = data[0]
        max_idx = np.argmax(first_arr)
        selected_point = np.unravel_index(max_idx, first_arr.shape)
    else:
        selected_point = points
    value_over_time = np.log1p(data[:, selected_point[0], selected_point[1]])
    plt.figure(figsize=(6, 4))

    plt.plot(
        range(len(value_over_time) - 1),
        value_over_time[:-1],
        marker='o',
        linestyle='-',
        color='b',
        label='Ground truth'
    )

    plt.plot(
        [len(value_over_time) - 1.95, len(value_over_time) - 1],
        [value_over_time[-2]-0.01, value_over_time[-1]],
        color='r',
        linestyle='-',
    )

    plt.scatter(
        len(value_over_time) - 1,
        value_over_time[-1],
        color='r',
        s=80,
        label='Predicted'
    )
    plt.legend()
    plt.title(f"Progress over time for point ({selected_point[0]}, {selected_point[1]})")
    plt.xlabel("t")
    plt.ylabel("log(1 + value)")
    plt.xticks(range(len(value_over_time)))
    plt.tight_layout()
    plt.show()
