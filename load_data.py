import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']  # shape: (10000, 3072)
        Y = datadict[b'labels']  # list of 10000 labels
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")  # 转换为 (N, 32, 32, 3)
        Y = np.array(Y)
        return X, Y


def load_cifar10(root):
    xs = []
    ys = []
    for i in range(1, 6):
        f = os.path.join(root, f'data_batch_{i}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)  # shape: (50000, 32, 32, 3)
    y_train = np.concatenate(ys)

    X_test, y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))  # shape: (10000, 32, 32, 3)
    return X_train, y_train, X_test, y_test


def preprocess_cifar10(X_train, y_train, X_test, y_test, val_split=0.1, flatten=True, normalize=True):
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0


    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)  # (N, 3072)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data(root, val_split=0.1, flatten=True, normalize=True):
    X_train, y_train, X_test, y_test = load_cifar10(root)
    return preprocess_cifar10(X_train, y_train, X_test, y_test, val_split, flatten, normalize)
