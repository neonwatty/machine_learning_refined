# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def PCA_demo():

    global n

    # load in data
    X = np.matrix(np.genfromtxt('PCA_demo_data.csv', delimiter=','))
    n = np.shape(X)[0]
    means = np.matlib.repmat(np.mean(X,0), n, 1)
    X = X - means  # center the data
    X = X.T
    K = 1

    # run PCA
    C, W = your_PCA(X, K)

    # plot results
    plot_results(X, C)
    plt.show()

    return


def your_PCA(X, K):

# ---> YOUR CODE GOES HERE.

    return C, W

def plot_results(X, C):

    # Print points and pcs
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    for j in np.arange(0,n):
        plt.scatter(X[0][:],X[1][:])

    s = np.arange(C[0,0],-C[0,0],.001)
    m = C[1,0]/C[0,0]
    ax1.plot(s, m*s, color = 'k', linewidth = 2)

    ax1.set_xlabel('$b_1$', fontsize = 14)
    ax1.set_ylabel('$b_2$', fontsize = 14)
    ax1.set_xlim(-.5, .5)
    ax1.set_ylim(-.5, .5)
    ax1.set_aspect('equal')

    # Plot projected data
    ax2 = fig.add_subplot(122)
    X_proj = np.dot(C, np.linalg.solve(np.dot(C.T,C),np.dot(C.T,X)))
    for j in np.arange(0,n):
        plt.scatter(X_proj[0][:],X_proj[1][:])

    ax2.set_xlabel('$b_1$', fontsize = 14)
    ax2.set_ylabel('$b_2$', fontsize = 14)
    ax2.set_xlim(-.5, .5)
    ax2.set_ylim(-.5, .5)
    ax2.set_aspect('equal')

    return

PCA_demo()
