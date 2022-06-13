# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division
import copy as cp


def ova_fixed_basis():

    colors = ['m','b','r','c']

    # parameters to play with
    deg = 2    # range of poly models to compare

    # load data
    X, y = load_data()
    num_classes = np.size(np.unique(y))  # number of classes = number of separators

    # make individual classifiers for each class
    ot = np.arange(0,1,.002)
    t1, t2 = np.meshgrid(ot,ot)
    t1 = np.reshape(t1,(np.size(t1),1))
    t2 = np.reshape(t2,(np.size(t2),1))
    X2 = np.concatenate((t1,t2),1)
    F = poly_features(X,deg)
    F2 = poly_features(X2,deg)

    for q in np.arange(1,num_classes+1):

        ind = np.nonzero(y == q)
        ind2 = np.nonzero(y != q)
        ytemp = cp.deepcopy(y)
        ytemp[ind] = 1
        ytemp[ind2] = -1

        w = log_loss_newton(F.T, ytemp)
        plt.subplot(1,num_classes+1,q)
        plot_poly(w,deg,colors[q])

        # calculate val
        u = np.dot(F2,w)

        if q ==1:
            M = u
        else:
            M = np.concatenate((M,u),1)

    z = np.argmax(M,1)

    # plot max separator on the whole thing
    t1 = np.reshape(t1,(np.size(ot),np.size(ot)))
    t2 = np.reshape(t2,(np.size(ot),np.size(ot)))
    z = np.reshape(z,(np.size(ot),np.size(ot)))

    plt.subplot(1,num_classes + 1,num_classes + 1)

    for i in np.arange(1,num_classes):
        plt.contour(t1,t2,z,2,color = 'k')


    plt.show()


### builds (poly) features based on input data ###
def poly_features(data, deg):

# ---> YOUR CODE GOES HERE.

    return F


### newton's method for log-loss classifier ###
def log_loss_newton(D,b):
    # initialize
    w = np.random.randn(np.shape(D)[0],1)

    # precomputations
    H = np.dot(np.diag(b),D.T)
    grad = 1
    n = 1

    ### main ###
    while (n <= 30) & (np.linalg.norm(grad) > 1e-5):

        # prep gradient for logistic objective
        r = sigmoid(-np.dot(H,w))
        g = r*(1 - r)
        grad = -np.dot(H.T,r)
        hess = np.dot(D,np.dot(np.diag(np.ravel(g)),D.T))

        # take Newton step
        s = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),s)
        n = n + 1

    return w


### sigmoid function for use with log_loss_newton ###
def sigmoid(z):
    return 1/(1+np.exp(-z))


### plots learned model ###
def plot_poly(w,deg,color):
    # Generate poly seperator
    o = np.arange(0,1,.01)
    s, t = np.meshgrid(o,o)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    f = np.zeros((np.size(s),1))
    count = 0

    for n in np.arange(0,deg+1):
        for m in np.arange(0,deg+1):
            if (n + m <= deg):
                f = f + w[count]*((s**n)*(t**m))
                count = count + 1

    s = np.reshape(s,(np.size(o),np.size(o)))
    t = np.reshape(t,(np.size(o),np.size(o)))
    f = np.reshape(f,(np.size(o),np.size(o)))

    # plot contour in original space
    plt.contour(s,t,f,1, colors =color)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('equal')


### load data ###
def load_data():
    # load data from file
    data = np.array(np.genfromtxt('bullseye_data.csv', delimiter=','))
    X = data[:,0:-1]
    y = data[:,-1]

    # how many classes in the data?  4 maximum for this toy.
    class_labels = np.unique(y)         # class labels
    num_classes = np.size(class_labels)

    fig = plt.figure(facecolor = 'white')
    colors = ['m','b','r','c']

    for j in np.arange(1,num_classes+1):
        # plot data
        plt.subplot(1,num_classes + 1,j)
        ind = np.nonzero(y != j)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = 'grey')
        ind = np.nonzero(y == j)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = colors[j])

        plt.subplot(1,num_classes + 1,num_classes + 1)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = colors[j])

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('equal')

    return X,y


ova_fixed_basis()
