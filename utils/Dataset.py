import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io


class Dataset():
    def __init__(self, name):
        self.path = './dataset/'
        self.name = name

    def load_data_3views(self):
        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)
        x1, x2, x3, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['gt']
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        return x1, x2, x3, y

    def load_data_4views(self):
        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)
        x1, x2, x3, x4, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], dataset['gt']
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        return x1, x2, x3, x4, y

    def load_data_6views(self):
        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)
        x1, x2, x3, x4, x5, x6, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], \
                                    dataset['x5'], dataset['x6'], dataset['gt']
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        return x1, x2, x3, x4, x5, x6, y

    def load_data_9views(self):
        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], \
                                                dataset['x5'], dataset['x6'], dataset['x7'], dataset['x8'], \
                                                dataset['x9'], dataset['gt']

        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        return x1, x2, x3, x4, x5, x6, x7, x8, x9, y

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        if 'ORL' in self.name or 'BBCSport' in self.name or 'AE2' in self.name or '2views' in self.name or 'dB' in self.name:
            dataset = scipy.io.loadmat(data_path)
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
        else:
            dataset = h5py.File(data_path, mode='r')
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            x1, x2, y = x1.value, x2.value, y.value
            x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
        return x1, x2, y

    def normalize(self, x, min=0):
        # min_val = np.min(x)
        # max_val = np.max(x)
        # x = (x - min_val) / (max_val - min_val)
        # return x

        if min == 0:
            scaler = MinMaxScaler([0, 1])
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x
