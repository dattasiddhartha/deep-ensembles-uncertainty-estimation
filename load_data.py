from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import os
import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import pickle


def load_data(path='notMNIST.pickle'):
    """
    Read a pre-prepared pickled dataset of the notmnist dataset
    (http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
    Args:
        path: Where to find the file, relative to .
    Returns: tuple of three pairs of np arrays
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
        The x arrays have shape (-1,28,28), and x values are floating point, normalized to have approximately zero mean
        and standard deviation ~0.5.
        The y arrays are single dimensional categorical (not 1-hot).
    """
    with open(os.path.expanduser(path), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        train_x = save['train_dataset']
        train_y = save['train_labels']
        valid_x = save['valid_dataset']
        valid_y = save['valid_labels']
        test_x = save['test_dataset']
        test_y = save['test_labels']
        del save  # hint to help gc free up memory

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def data_notMNIST():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data('data/notMNIST.pickle')
    len(x_train), len(x_valid), len(x_test)

    # Reshape inputs to flat vectors, convert labels to one-hot.
    x_train = x_train.reshape(-1, 784)
    x_valid = x_valid.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'valid samples')
    print(x_test.shape[0], 'test samples')

    batch_size = 128
    nb_classes = 10
    nb_epoch = 10

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# load MNIST



def load_mnist(path='MNIST.pickle'):
    with open(os.path.expanduser(path), 'rb') as f:
        train, val, test = pickle.load(f, encoding='latin1')
#         del save  # hint to help gc free up memory

        return train, val, test


def data_MNIST():
    train, val, test = load_mnist("data/mnist.pkl")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train[0], train[1], val[0], val[1], test[0], test[1]
    ## Changing dimension of input images from N*28*28 to  N*784
    # X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    # X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    # print('Train dimension:');print(X_train.shape)
    # print('Test dimension:');print(X_test.shape)
    ## Changing labels to one-hot encoded vector
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)
    print('Train labels dimension:');print(Y_train.shape)
    print('Test labels dimension:');print(Y_test.shape)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test