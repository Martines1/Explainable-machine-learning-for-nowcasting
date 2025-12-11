import sys
import numpy as np
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
        utils.show_and_save_mask(frames[i], masks[i], f'diff_channel_{i}')
    return masks
