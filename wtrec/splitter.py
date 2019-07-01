from random import sample
import numpy as np
import os


Y_COLUMN = 'label'
DROP_COLUMS = ['raw', 'path']


def _shuffle(data):
    return data.reindex(np.random.permutation(data.index))


def split_random(data, test_size=0.3):
    test_size = int(data.shape[0] * test_size)
    train_size = len(data) - test_size 

    data = _shuffle(data)[:test_size + train_size]
    data = data.reindex().drop(DROP_COLUMS, axis=1)

    test = data[:test_size]
    train = data[test_size:]

    test_Y = test[Y_COLUMN]
    test_X = test.drop(Y_COLUMN, axis=1)

    train_Y = train[Y_COLUMN]
    train_X = train.drop(Y_COLUMN, axis=1)

    return train_X, train_Y, test_X, test_Y


def _train_test_indices(num_samples, idx_file_path, test_size=0.3):
    if not os.path.isfile(idx_file_path):
        test_indices = np.array(sample(range(num_samples), k=int(num_samples * test_size)))
        test_indices.dump(idx_file_path)

    test_indices = np.load(idx_file_path)
    train_indices = np.array([i for i in range(num_samples) if i not in test_indices])

    return train_indices, test_indices


def split_fixed(data, idx_file_path):
    data = data.reindex().drop(DROP_COLUMS, axis=1)

    train_indices, test_indices = _train_test_indices(data.shape[0], idx_file_path)

    test = data.iloc[test_indices]
    train = data.iloc[train_indices]

    test_Y = test[Y_COLUMN]
    test_X = test.drop(Y_COLUMN, axis=1)

    train_Y = train[Y_COLUMN]
    train_X = train.drop(Y_COLUMN, axis=1)

    return train_X, train_Y, test_X, test_Y
