# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def K_means_demo():

    # load data
    X = np.array(np.genfromtxt('Kmeans_demo_data.csv', delimiter=','))

    C0 = np.array([[0,0],[0,.5]])   # initial centroid locations

    # run K-means
    K = np.shape(C0)[1]

    C, W = your_K_means(X, K, C0)

    # plot results
    plot_results(X, C, W, C0)
    plt.show()


def your_K_means(X, K, C0):

# --->  YOUR CODE GOES HERE.

    return C, W

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')

K_means_demo()
