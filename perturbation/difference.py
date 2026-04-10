import sys
import numpy as np
import matplotlib.pyplot as plt
from rainnet import utils


def calculate_diff_rain_appear(frame1, frame2, no_precip_value):
    if frame1.shape != frame2.shape:
        sys.exit("frames does not have same dimensions!")
    return (frame1 == no_precip_value) & (frame2 > no_precip_value)


def calculate_diff_rain_disappear(frame1, frame2, no_precip_value):
    if frame1.shape != frame2.shape:
        sys.exit("frames does not have same dimensions!")
    return (frame1 > no_precip_value) & (frame2 == no_precip_value)


def calculate_diff_both(frame1, frame2, no_precip_value):
    if frame1.shape != frame2.shape:
        sys.exit("frames does not have same dimensions!")
    appear_mask = (frame1 == no_precip_value) & (frame2 > no_precip_value)
    disappear_mask = (frame1 > no_precip_value) & (frame2 == no_precip_value)
    return appear_mask | disappear_mask


def calculate_diff_unique(frames, no_precip_value):
    if frames.ndim != 3:
        sys.exit("frames must have shape (C, H, W)!")
    frames = np.asarray(frames, dtype=np.float64)
    C, H, W = frames.shape
    frames = np.round(frames, decimals=3)
    if C < 2:
        sys.exit("At least two frames are required to calculate differences!")

    masks = np.zeros((C, H, W), dtype=bool)
    for i in range(C):
        masks[i] = frames[i] > no_precip_value
        for j in range(C):
            if i == j:
                continue
            masks[i] &= (frames[j] == no_precip_value)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=100)
    fig.suptitle('Global unique points', fontsize=24, fontweight='bold', y=0.98)

    axes_flat = axes.flatten()
    h, w = frames[0].shape

    for i in range(C):
        ax = axes_flat[i]
        rgba = utils.cmap(utils.norm(frames[i]))
        rgba[..., 3] = 0.2
        rgba[masks[i]] = [1.0, 0.0, 0.0, 1.0]

        ax.imshow(
            rgba,
            interpolation='nearest',
            origin='upper',
            extent=[0, w, 0, h]
        )

        ax.set_title(f'Channel {i + 1}', fontsize=18, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=16)
        ax.set_ylabel('Y (pixels)', fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=1.0, label='Unique point')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    from pathlib import Path
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    plt.savefig("output/clean/global_unique_points.png", dpi=100, bbox_inches='tight')
    plt.close()

    return masks


def calculate_diff_unique_single(frames, no_precip_value, selected_frame):
    """
    Alternative to calculate_diff_unique that only compares one frame to the others.
    """

    if frames.ndim != 3:
        sys.exit("frames must have shape (C, H, W)!")
    frames = np.asarray(frames, dtype=np.float64)
    C, H, W = frames.shape
    frames = np.round(frames, decimals=3)
    if C < 2:
        sys.exit("At least two frames are required to calculate differences!")

    masks = np.zeros((C, H, W), dtype=bool)
    masks[selected_frame] = frames[selected_frame] > no_precip_value
    for j in range(C):
        if selected_frame == j:
            continue
        masks[selected_frame] &= (frames[j] == no_precip_value)
    return masks[selected_frame]


def compare_all(X_raw, no_precip_value, selected_channel):
    selected_channel = selected_channel - 1  # Convert to 0-based index
    if selected_channel < 1 or selected_channel > X_raw.shape[0] - 2:
        max_cnt = X_raw.shape[0] - 2
        sys.exit(f"selected_channel must be between <1, {max_cnt}>!")

    selected_frame = X_raw[selected_channel]

    appear = calculate_diff_rain_appear(X_raw[selected_channel - 1], X_raw[selected_channel], no_precip_value)
    disappear = calculate_diff_rain_disappear(X_raw[selected_channel], X_raw[selected_channel + 1], no_precip_value)
    both = appear | disappear
    global_unique = calculate_diff_unique_single(X_raw, no_precip_value, selected_channel)

    cnt_unique_ratio = (np.sum(global_unique) / np.sum(both)) if np.sum(both) > 0 else None
    print("Global unique points count: ", np.sum(global_unique))
    print("Both (appear + disappear) points count: ", np.sum(both))
    print("Unique points count in both uniqueness: ", np.sum(both) - np.sum(global_unique))
    print(f"Unique points ratio in both uniqueness: {cnt_unique_ratio}")
    print(np.all(global_unique <= both))
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=100)

    title = f'Comparison of uniqueness - channel {selected_channel + 1}'
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

    masks_data = [
        (appear, 'Local rain appear'),
        (disappear, 'Local rain disappear'),
        (both, 'Local rain both (appear + disappear)'),
        (global_unique, 'Global unique')
    ]

    axes_flat = axes.flatten()
    h, w = selected_frame.shape
    for ax, (mask, mask_title) in zip(axes_flat, masks_data):
        rgba = utils.cmap(utils.norm(selected_frame))
        rgba[..., 3] = 0.2
        rgba[mask] = [1.0, 0.0, 0.0, 1.0]

        ax.imshow(
            rgba,
            interpolation='nearest',
            origin='upper',
            extent=[0, w, 0, h]
        )

        ax.set_title(mask_title, fontsize=18, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=16)
        ax.set_ylabel('Y (pixels)', fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)

        xticks = list(np.arange(0, w, 200))
        yticks = list(np.arange(0, h, 200))

        if xticks[-1] != w:
            xticks.append(w)
        if yticks[-1] != h:
            yticks.append(h)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=1.0, label='Unique point')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    from pathlib import Path
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/clean").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"output/clean/compare_all_channel_{selected_channel}.png", dpi=100, bbox_inches='tight')
    plt.close()

    return {'appear': appear, 'disappear': disappear, 'both': both, 'global_unique': global_unique}
