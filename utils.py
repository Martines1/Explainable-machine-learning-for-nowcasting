import numpy as np

def Scaler(array):
    return np.log(array + 0.01)


def invScaler(array):
    return np.exp(array) - 0.01


def pad_to_shape(array, from_shape=900, to_shape=928, how="mirror"):
    # calculate how much to pad in respect with native resolution
    padding = int((to_shape - from_shape) / 2)
    # for input shape as (batch, W, H, channels)
    if how == "zero":
        array_padded = np.pad(array, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant",
                              constant_values=0)
    elif how == "mirror":
        array_padded = np.pad(array, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="reflect")
    return array_padded


def pred_to_rad(pred, from_shape=928, to_shape=900):
    # pred shape 12,928,928
    padding = int((from_shape - to_shape) / 2)
    return pred[::, padding:padding + to_shape, padding:padding + to_shape].copy()


def data_preprocessing(X):
    # 0. Right shape for batch
    X = np.moveaxis(X, 0, -1)
    X = X[np.newaxis, ::, ::, ::]
    # 1. To log scale
    X = Scaler(X)
    # 2. from 900x900 to 928x928
    X = pad_to_shape(X)

    return X


def data_postprocessing(nwcst):
    # 0. Squeeze empty dimensions
    nwcst = np.squeeze(np.array(nwcst))
    # 1. Convert back to rainfall depth
    nwcst = invScaler(nwcst)
    # 2. Convert from 928x928 back to 900x900
    nwcst = pred_to_rad(nwcst)
    # 3. Return only positive values
    nwcst = np.where(nwcst > 0, nwcst, 0)
    return nwcst