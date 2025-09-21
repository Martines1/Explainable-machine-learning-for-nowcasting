import tensorflow as tf
import numpy as np
from utils import data_preprocessing


class Dataset(tf.keras.utils.Sequence):

    def __init__(
            self,
            dataset_dict,
            image_names,
            batch_size
    ):
        self.keys = image_names
        self.dataset = dataset_dict
        self.bs = batch_size

    def get_index(self, i):
        x = []
        for j in range(4):
            try:
                arr = np.array(self.dataset.get(self.keys[i + j]))
            except:
                print(i, j)
            x.append(arr)

        x = data_preprocessing(np.stack(x, 0))
        # x = np.transpose(np.squeeze(x),(2,0,1))
        x = np.squeeze(x)
        y = np.squeeze(data_preprocessing(np.array(self.dataset[self.keys[i + 3]])[np.newaxis, :, :]))

        return x.astype('float32'), y.astype('float32')

    def __getitem__(self, index):

        X = []
        Y = []

        for i in range(index * self.bs, (index + 1) * self.bs):
            x, y = self.get_index(i)
            X.append(x[np.newaxis, :])
            Y.append(y[np.newaxis, :])

        return X, Y

    def __len__(self):
        return (len(self.keys) - 4) // self.bs