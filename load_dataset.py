#  -*- coding: utf-8 -*-
#  Author : Subin Lee
#  e-mail : subin.lee@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
"""
load dataset
"""
from keras.datasets import cifar10
from keras.utils import to_categorical


def load_dataset():
    """cifar10 dataset"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    normalize = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train /= 255.0
    x_test /= 255.0

    mean = normalize[0]
    std = normalize[1]

    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
