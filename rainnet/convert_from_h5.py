import h5py
import numpy as np
import torch
from rainnet_arch import RainNet

KERAS_CONV_NODE_FMT = "model_weights/conv2d_{i}/conv2d_{i}/"


def load_keras_h5_into_torch(h5_path: str, in_channels=4) -> RainNet:
    model = RainNet(in_channels=in_channels).eval()
    convs = model.convs_in_keras_order()

    with h5py.File(h5_path, "r") as f:
        for idx, conv in enumerate(convs, start=1):
            base = KERAS_CONV_NODE_FMT.format(i=idx)
            k_name = base + "kernel:0"
            b_name = base + "bias:0"

            k_tf = f[k_name][()]
            b_tf = f[b_name][()]

            k_pt = np.transpose(k_tf, (3, 2, 0, 1)).astype(np.float64)
            b_pt = b_tf.astype(np.float64)

            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy(k_pt))
                conv.bias.copy_(torch.from_numpy(b_pt))

    return model
