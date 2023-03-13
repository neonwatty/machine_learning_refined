# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def compare_maps_regression_hw():

    # load in data
    data = np.array(np.genfromtxt('noisy_sin_samples.csv', delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))

    # true underlying data-generating function
    global x_true,y_true
    x_true = np.reshape(np.arange(0,1,.01),(100,1))
    y_true = np.sin(2*np.pi*x_true)

    # parameters to play with
    k = 3    # number of folds to use

    # split points into k equal sized sets
    c = split_data(x,y,k)

    ###################################################
    # do k-fold cross-validation using polynomial basis
    poly_degs = np.arange(1,11)           # range of poly models to compare
    deg = cross_validate_poly(x,y,c,poly_degs,k)


    # plot it
    plt.subplot(1,3,1)
    plot_poly(x,y,deg,k)

    ###################################################
    # do k-fold cross-validation using fourier basis
    fourier_degs = np.arange(1,11)           # range of fourier models to compare
    deg = cross_validate_fourier(x,y,c,fourier_degs,k)


    # plot it
    plt.subplot(1,3,2)
    plot_fourier(x,y,deg,k)

    ###################################################
    # do k-fold cross-validation using tanh basis
    tanh_degs = np.arange(1,7)           # range of NN models to compare
    deg = cross_validate_tanh(x,y,c,tanh_degs,k)

    # plot it
    plt.subplot(1,3,3)
    plot_tanh(x,y,deg,k)
    plt.show()


def split_data(x,y,k):

# ---> YOUR CODE GOES HERE.

    return c

def cross_validate_poly(x,y,c,poly_degs,k):

# ---> YOUR CODE GOES HERE.

    return deg

def cross_validate_fourier(x,y,c,fourier_degs,k):

# ---> YOUR CODE GOES HERE.

    return deg

def cross_validate_tanh(x,y,split,tanh_degs,k):

# ---> YOUR CODE GOES HERE.

    return deg


#########################################

### takes poly features of the input ###
def build_poly(x,D):
    F = []
    for m in np.arange(1,D+1):
        F.append(x**m)

    temp1 = np.reshape(F,(D,np.shape(x)[0])).T
    temp2 = np.concatenate((np.ones((np.shape(temp1)[0],1)),temp1),1)
    F = temp2
    return F

### takes fourier features of the input ###
def build_fourier(x,D):
    F = []
    for m in np.arange(1,D+1):
        F.append(np.cos(2*np.pi*m*x))
        F.append(np.sin(2*np.pi*m*x))

    temp1 = np.reshape(F,(2*D,np.shape(x)[0])).T
    temp2 = np.concatenate((np.ones((np.shape(temp1)[0],1)),temp1),1)
    F = temp2
    return F

### gradient descent for single layer tanh nn ###
def tanh_grad_descent(x,y,M):

    # initializations
    P = np.size(x)
    b = M*np.random.randn()
    w = M*np.random.randn(M,1)
    c = M*np.random.randn(M,1)
    v = M*np.random.randn(M,1)
    l_P = np.ones((P,1))

    # stoppers + step length
    max_its = 10000
    k = 1
    alpha = 1e-4

    ### main ###
    while (k <= max_its):
        # update gradients
        q = np.zeros((P,1))
        for p in np.arange(0,P):
            q[p] = b + np.dot(w.T,np.tanh(c + v*x[p])) - y[p]

        grad_b = np.dot(l_P.T,q)
        grad_w = np.zeros((M,1))
        grad_c = np.zeros((M,1))
        grad_v = np.zeros((M,1))
        for m in np.arange(0,M):
            t = np.tanh(c[m] + x*v[m])
            s = (1/np.cosh(c[m] + x*v[m]))**2
            grad_w[m] = 2*np.dot(l_P.T,(q*t))
            grad_c[m] = 2*np.dot(l_P.T,(q*s)*w[m])
            grad_v[m] = 2*np.dot(l_P.T,(q*x*s)*w[m])

        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        v = v - alpha*grad_v

        # update stopper and container
        k = k + 1

    return b, w, c, v

def plot_poly(x,y,deg,k):

    # calculate weights
    X = build_poly(x,deg)
    temp = np.linalg.pinv(np.dot(X.T,X))
    w = np.dot(np.dot(temp,X.T),y)

    # output model
    in_put = np.reshape(np.arange(0,1,.01),(100,1))
    out_put = np.zeros(np.shape(in_put))
    for n in np.arange(0,deg+1):
        out_put = out_put + w[n]*(in_put**n)

    # plot
    plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
    plt.scatter(x,y,s=40,color = 'k')
    plt.plot(in_put,out_put,'b',linewidth=2)

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

def plot_fourier(x,y,deg,k):

    # calculate weights
    X = build_fourier(x,deg)
    temp = np.linalg.pinv(np.dot(X.T,X))
    w = np.dot(np.dot(temp,X.T),y)

    # output model
    period = 1
    in_put = np.reshape(np.arange(0,1,.01),(100,1))
    out_put = w[0]*np.ones(np.shape(in_put))
    for n in np.arange(1,deg+1):
        out_put = out_put + w[2*n-1]*np.cos((1/period)*2*np.pi*n*in_put)
        out_put = out_put + w[2*n]*np.sin((1/period)*2*np.pi*n*in_put)

    # plot
    plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
    plt.scatter(x,y,s=40,color = 'k')
    plt.plot(in_put,out_put,'r',linewidth=2)

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

def plot_tanh(x,y,deg,k):

    # calculate weights
    colors = ['m','c']
    num_inits = 2
    for foo in np.arange(0,num_inits):
        b, w, c, v = tanh_grad_descent(x,y,deg)

        # plot resulting fit
        plot_approx(b,w,c,v,colors[foo])

    # plot
    plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
    plt.scatter(x,y,s=40,color = 'k')

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

### plot tanh approximation ###
def plot_approx(b,w,c,v,color):
    M = np.size(c)
    s = np.arange(0,1,.01)
    t = b
    for m in np.arange(0,M):
        t = t + w[m]*np.tanh(c[m] + v[m]*s)

    s = np.reshape(s,np.shape(t))
    plt.plot(s[0],t[0], color = color, linewidth=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

compare_maps_regression_hw()
