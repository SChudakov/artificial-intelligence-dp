#!/usr/bin/env python3
import sys
import time
import numpy as np


def get_mini_batches(data, mini_batch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for mini_batch_start in np.arange(0, data_size, mini_batch_size):
        mini_batch_indices = indices[mini_batch_start:mini_batch_start + mini_batch_size]
        yield [_mini_batch(d, mini_batch_indices) for d in data] if list_data \
            else _mini_batch(data, mini_batch_indices)


def _mini_batch(data, mini_batch_idx):
    return data[mini_batch_idx] if type(data) is np.ndarray else [data[i] for i in mini_batch_idx]


def test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print(name, "passed!")
