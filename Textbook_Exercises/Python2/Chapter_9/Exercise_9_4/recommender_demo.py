# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def recommender_demo():
    # load in data
    X = np.array(np.genfromtxt('recommender_demo_data_true_matrix.csv', delimiter=','))
    X_corrupt = np.array(np.genfromtxt('recommender_demo_data_dissolved_matrix.csv', delimiter=','))

    K = np.linalg.matrix_rank(X)

    # run ALS for matrix completion
    C, W = matrix_complete(X_corrupt, K)

    # plot results
    plot_results(X, X_corrupt, C, W)
    plt.show()


def matrix_complete(X, K):

# ---->  YOUR CODE GOES HERE
    
    return C, W

def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap='Greys_r')
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap='Greys_r')
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap='Greys_r')
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)

recommender_demo()
