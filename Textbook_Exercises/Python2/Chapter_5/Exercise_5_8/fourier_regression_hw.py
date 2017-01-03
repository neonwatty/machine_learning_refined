# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division
import pylab

def fourier_regression_hw():

    # load data
    x, y = load_data()
    deg = [1,3,5,7,9,15]           # degrees to try

    # plot data
    plot_data(x,y,deg)

    # generate nonlinear features
    mses = []

    for D in np.arange(0,np.size(deg)):
        # generate poly feature transformation
        F = fourier_features(x,deg[D])

        # get weights
        temp = np.linalg.pinv(np.dot(F,F.T))
        w = np.dot(np.dot(temp,F),y)
        MSE = np.linalg.norm(np.dot(F.T,w)-y)/np.size(y)
        mses.append(MSE)

        # plot fit to data
        plt.subplot(2,3,D+1)
        plot_model(w,deg[D])


    # make plot of mse's
    plot_mse(mses,deg)
    plt.show()

### takes fourier features of the input ###
def fourier_features(x,D):

# ----->  YOUR CODE GOES HERE.

    return F

def load_data():
    data = np.array(np.genfromtxt('noisy_sin_samples.csv', delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))
    return x,y


## plot the D-fit ###
def plot_model(w,D):

    # plot determined surface in 3d space
    s = np.arange(0,1,.01)
    f = []
    for m in np.arange(1,D+1):
        f.append(np.cos(2*np.pi*m*s))
        f.append(np.sin(2*np.pi*m*s))

    f = np.reshape(f,(2*D,np.size(s))).T
    temp = np.dot(f,np.reshape(w[1:],(np.size(w)-1,1)))

    f = np.sum(temp,1) + w[0]

    # plot contour in original space
    plt.plot(s,f, color = 'r', linewidth = 2)
    plt.ylim(-1.5,1.5)
    plt.xlim(0,1)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

### plot mse's over all D tested ###
def plot_mse(mses,deg):
    plt.figure(2)
    plt.plot(deg,mses,'ro--')
    plt.title('MSE on entire dataset in D', fontsize=18)
    plt.xlabel('D', fontsize=18)
    plt.ylabel('MSE       ', fontsize=18)


### plot data ###
def plot_data(x,y,deg):
    for i in np.arange(1,7):
        plt.subplot(2,3,i)
        plt.scatter(x,y,s = 30, color = 'k')

        # graph info labels
        s = 'D = ' + str(deg[i-1])
        plt.title(s, fontsize=15)

fourier_regression_hw()
