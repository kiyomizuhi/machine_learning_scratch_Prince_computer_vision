#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:02:20 2017

@author: hiroyukiinoue
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def __main__():
    import numpy as np

    locationM = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/batches.meta' 
    location1 = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/data_batch_1'
    location2 = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/data_batch_2'
    location3 = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/data_batch_3'
    location4 = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/data_batch_4'
    location5 = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/data_batch_5'
    locationT = '/Users/hiroyukiinoue/Documents/Python_Code/CS231/cifar-10-batches-py/test_batch'

    data1 = unpickle(location1)
    data2 = unpickle(location2)
    data3 = unpickle(location3)
    data4 = unpickle(location4)
    data5 = unpickle(location5)
    dataT = unpickle(locationT)

    Xtr = np.zeros((50000, 3072), dtype=int)
    Xtr[0:10000,:] = data1[b'data']
    Xtr[10000:20000,:] = data2[b'data']
    Xtr[20000:30000,:] = data3[b'data']
    Xtr[30000:40000,:] = data4[b'data']
    Xtr[40000:50000,:] = data5[b'data']

    ytr = np.zeros(50000, dtype=int)
    ytr[0:10000] = np.array(data1[b'labels'])
    ytr[10000:20000] = np.array(data2[b'labels'])
    ytr[20000:30000] = np.array(data3[b'labels'])
    ytr[30000:40000] = np.array(data4[b'labels'])
    ytr[40000:50000] = np.array(data5[b'labels'])
    
    Xte = dataT[b'data'] 
    yte = np.array(dataT[b'labels'])
    
    return Xtr, ytr, Xte, yte

def view(x):
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.imshow(x.reshape(3, 32, 32).transpose(1,2,0).astype("uint8"))