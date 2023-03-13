# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def tanh_regression_hw():

    # load data
    x, y = load_data()

    # plot data
    plt.subplot(1,2,1)
    plot_data(x,y)

    # perform gradient descent to fit tanh basis sum
    num_runs = 3
    colors = ['r','g','b']
    for i in np.arange(0,num_runs):

        # minimize least squares cost
        b,w,c,v,obj_val = tanh_grad_descent(x,y,i)

        # plot resulting fit to data
        color = colors[i]
        plt.subplot(1,2,1)
        plot_approx(b,w,c,v,color)

        # plot objective value decrease for current run
        plt.subplot(1,2,2)
        plot_obj(obj_val,color)

    plt.show()

### gradient descent for single layer tanh nn ###
def tanh_grad_descent(x,y,i):

    # initialize weights and other items
    b, w, c, v = initialize(i)
    P = np.size(x)
    M = 4
    alpha = 1e-3
    l_P = np.ones((P,1))

    # stoppers and containers
    max_its = 15000
    k = 1
    obj_val = []       # container for objective value at each iteration

    ### main ###
    while (k <= max_its):
        # update gradients
# --->  grad_b =
# --->  grad_w =
# --->  grad_c =
# --->  grad_v =

        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        v = v - alpha*grad_v

        # update stopper and container
        k = k + 1
        obj_val.append(calculate_obj_val(x,y,b,w,c,v))

    return b, w, c, v, obj_val


def load_data():
    data = np.array(np.genfromtxt('noisy_sin_samples.csv', delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))
    return x,y


def calculate_obj_val(x,y,b,w,c,v):
    s = 0
    P = np.size(x)
    for p  in np.arange(0,P):
        s = s + ((b + np.dot(w.T,np.tanh(c + v*x[p])) - y[p])**2)
    return s[0][0]

### initialize parameters ###
def initialize(i):
    b = 0
    w = 0
    c = 0
    v = 0
    if (i == 0):
        b = -0.454
        w = np.array([[-0.3461],[-0.8727],[0.6312 ],[0.9760]])
        c = np.array([[-0.6584],[0.7832],[-1.0260],[0.5559]])
        v = np.array([[-0.8571],[-0.8623],[1.0418],[-0.4081]])

    elif (i == 1):
        b = -1.1724
        w = np.array([[.09],[-1.99],[-3.68],[-.64]])
        c = np.array([[-3.4814],[-0.3177],[-4.7905],[-1.5374]])
        v = np.array([[-0.7055],[-0.6778],[0.1639],[-2.4117]])

    else:
        b = 0.1409
        w = np.array([[0.5207],[-2.1275],[10.7415],[3.5584]])
        c = np.array([[2.7754],[0.0417],[-5.5907],[-2.5756]])
        v = np.array([[-1.8030],[0.7578],[-2.4235],[0.6615]])

    return b, w, c, v


### plot tanh approximation ###
def plot_approx(b,w,c,v,color):
    M = np.size(c)
    s = np.arange(0,1,.01)
    t = b
    for m in np.arange(0,M):
        t = t + w[m]*np.tanh(c[m] + v[m]*s)

    s = np.reshape(s,np.shape(t))
    plt.plot(s[0],t[0], color = color, linewidth=2)
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$  ', fontsize=20)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

### plot objective value at each iteration of gradient descent ###
def plot_obj(o, color):
    if np.size(o) == 15000:
        plt.plot(np.arange(100,np.size(o)), o[100:], color = color, linewidth=2)
    else:
        plt.plot(np.arange(1,np.size(o)+1), o, color = color)

    plt.xlabel('$k$', fontsize=20)
    plt.ylabel('$g(w^k)$  ', fontsize=20)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


### plot data ###
def plot_data(x,y):
    plt.scatter(x,y,s=30,color='k')

tanh_regression_hw()
