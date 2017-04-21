# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:15:19 2016

@author: Ying Zhou
@PID: A53103642
@Email: yiz251@ucsd.edu
"""

import os
import struct
import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from random import randint


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

images, labels = load_mnist(dataset="training", digits = np.arange(10), path = ".")
test_images, test_labels = load_mnist(dataset = "testing", digits = np.arange(10), path = ".")

def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

M = 10000
######################## uniform random selection ########################
import random
rand = random.sample(range(0, 60000), M)
rand_images = images[rand]
rand_labels = labels[rand]
 
size = len(rand_images)
twodim_rand_images = rand_images.reshape(size,-1)
twodim_test_images = test_images.reshape(len(test_images),-1)

from sklearn.neighbors import KNeighborsClassifier

oneNN = KNeighborsClassifier(n_neighbors=1)
oneNN.fit(twodim_rand_images, rand_labels) 
result = oneNN.predict(twodim_test_images)
accur = oneNN.score(twodim_test_images, test_labels)
     
######################## prototype selection ###########################

########################### adaboost method ############################
'''rand is the index of test_images'''
def fault_idx(result, test_labels, accur, rand, size, K):
    result = [None]*((size/K)*(1-accur))
    j = 0
    for i in range(len(result)):
        if result[i] != test_labels[i]:
            result[j] = rand[i]
            j = j + 1
    return result

import random

def adaboost(size, K):
    rand = random.sample(range(0, 60000), size)
    ada_images = images[rand]
    ada_labels = labels[rand]
    ada_images = ada_images.reshape(len(ada_images), -1)
    fault = []
    for i in range(K):
        test_idx = rand[int(i*size/K):int((i+1)*size/K)]
        test_images = ada_images[int(i*size/K):int((i+1)*size/K)]
        test_labels = ada_labels[int(i*size/K):int((i+1)*size/K)]
        if i == 0:
            train_images = ada_images[int((i+1)*size/K):]
            train_labels = ada_labels[int((i+1)*size/K):]
        if i == K-1:
            train_images = ada_images[:int(i*size/K)]
            train_labels = ada_labels[:int(i*size/K)]
        else:
            train_images = np.concatenate((ada_images[:int(i*size/K)],ada_images[int((i+1)*size/K):]))
            train_labels = np.concatenate((ada_labels[:int(i*size/K)],ada_labels[int((i+1)*size/K):]))
        oneNN = KNeighborsClassifier(n_neighbors=1)
        oneNN.fit(train_images, train_labels) 
        result = oneNN.predict(test_images)
        accur = oneNN.score(test_images, test_labels)
        fault_add = fault_idx(result, test_labels, accur, test_idx, size, K)
        fault = np.concatenate((fault, fault_add))
    return fault

fault = adaboost(10000, 5)

##### add some random elements to make M to be 1000 #####
rand = random.sample(range(0, 60000), 5000-len(fault))
prototype = np.concatenate((fault, rand))
prototype = prototype.tolist()
proto_images = images[prototype]
proto_images = proto_images.reshape(len(proto_images), -1)
proto_labels = labels[prototype]

# check the accuracy #
oneNN = KNeighborsClassifier(n_neighbors=1)
oneNN.fit(proto_images, proto_labels) 
result = oneNN.predict(twodim_test_images)
accur = oneNN.score(twodim_test_images, test_labels)

############################# end adaboost ##############################

############################ CNN prototype ##############################
'''images: training images; labels: training labels; M: prototype size'''
def prototype(images, labels, M):
    prototype = []
    prototype.append(0)
    rand = random.sample(range(0,60000), 60000)
    for i in range(60000):
        if len(prototype) >= M:
            break
        label = find_label(prototype, rand[i])
        if label != labels[rand[i]]:
            prototype.append(rand[i])
    return prototype


from sklearn.neighbors import KNeighborsClassifier

'''prototype: prototype set indexes; i: the ith element of training set'''
def find_label(prototype, i):
    proto_images = images[prototype].reshape(len(prototype),-1)
    test_image = images[i].reshape(1,-1)
    oneNN = KNeighborsClassifier(n_neighbors=1)
    oneNN.fit(proto_images, labels[prototype].ravel())
    result = oneNN.predict(test_image)
    return result

# check the accuracy of prototype with size 1000
proto_1000 = prototype(images, labels, 1000)
proto_images = images[proto_1000]
proto_images = proto_images.reshape(len(proto_images), -1)
proto_labels = labels[proto_1000]
oneNN = KNeighborsClassifier(n_neighbors=1)
oneNN.fit(proto_images, proto_labels) 
twodim_test_images = test_images.reshape(len(test_images),-1)
# result = oneNN.predict(twodim_test_images)
accur = oneNN.score(twodim_test_images, test_labels) # 0.8903,0.8893, 0.8929, random is 0.8795, 0.8890, 0.8901

# check the accuracy of prototype with size 5000
proto_5000 = prototype(images, labels, 5000)
proto_images = images[proto_5000]
proto_images = proto_images.reshape(len(proto_images),-1)
proto_labels = labels[proto_5000]
oneNN = KNeighborsClassifier(n_neighbors=1)
oneNN.fit(proto_images, proto_labels) 
twodim_test_images = test_images.reshape(len(test_images),-1)
accur = oneNN.score(twodim_test_images, test_labels) # 0.939, random is 0.935

# check the accuracy of prototype with size 10000
proto_10000 = prototype(images, labels, 10000)
proto_images = images[proto_10000]
proto_images = proto_images.reshape(len(proto_images),-1)
proto_labels = labels[proto_10000]
oneNN = KNeighborsClassifier(n_neighbors=1)
oneNN.fit(proto_images, proto_labels) 
twodim_test_images = test_images.reshape(len(test_images),-1)
accur = oneNN.score(twodim_test_images, test_labels)

import numpy as np
import matplotlib.pyplot as plt
protomean = (0.1088, 0.0595, 0.0417)
randommean = (0.1149, 0.0655, 0.0564)
protostd = (0.001757,0.002042,0.002145)
randomstd = (0.00494, 0.002432, 0.00275) 
N = len(randommean)
ind = np.arange(N)
width = 0.4
fig,ax = plt.subplots()
rects1 = ax.bar(ind, protomean, width, color = 'MediumSlateBlue', yerr = protostd, error_kw = {'ecolor':'Tomato','linewidth':2})
rects2 = ax.bar(ind+width, randommean, width, color = 'Tomato', yerr = randomstd, error_kw = {'ecolor':'MediumSlateBlue','linewidth':2})
axes = plt.gca()
axes.set_ylim([0, 0.15]) 

ax.set_ylabel('Error Rate')
ax.set_title('Error Bar')
ax.set_xticks(ind+width)
ax.set_xticklabels(('M = 1000', 'M = 5000', 'M = 10000'))
ax.legend([rects1, rects2],['Prototype', 'Random'])









