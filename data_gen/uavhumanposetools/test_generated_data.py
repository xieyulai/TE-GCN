import pickle
path = 'data/Skeleton/train_label.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
    print(data[0][0])
    print(data[1][0])

import numpy as np
path = 'data/Skeleton/train_data.npy'
x = np.load(path)
print(x.shape)


path = 'data/Skeleton/test_label.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
    print(data[0][0])
    print(data[1][0])

import numpy as np
path = 'data/Skeleton/test_data.npy'
x = np.load(path)
print(x.shape)
