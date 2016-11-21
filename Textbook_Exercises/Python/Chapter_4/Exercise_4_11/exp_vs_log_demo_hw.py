# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import matplotlib.pyplot as plt
import numpy as np

# load the data
def load_data():
    # load data
    data = np.matrix(np.genfromtxt('exp_vs_log_data.csv', delimiter=','))
    X = np.asarray(data[:,0:2])
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    return X, y

###### ML Algorithm functions ######

# run gradient descent for h1
def gradient_descent_soft_cost(X,y,w,alpha):

    # start gradient descent loop
    grad = 1
    k = 1
    max_its = 10000

    while np.linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
# --->  grad =

        # take gradient step
        w = w - alpha*grad

        # update path containers
        k += 1

    return w


# run gradient descent for h2
def gradient_descent_exp_cost(X,y,w,alpha):

    # start gradient descent loop
    grad = 1
    k = 1
    max_its = 10000

    while np.linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
# --->  grad =

        # take gradient step
        w = w - alpha*grad

        # update path containers
        k += 1

    return w

###### plotting functions #######
def plot_all(X,y,w,color,ax1):

    # initialize figure, plot data, and dress up panels with axes labels etc.,

    ax1.set_xlabel('$x_1$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 20)
    s = np.argwhere(y == 1)
    s = s[:,0]
    plt.scatter(X[s,0],X[s,1], s = 30,color = (1, 0, 0.4))
    s = np.argwhere(y == -1)
    s = s[:,0]
    plt.scatter(X[s,0],X[s,1],s = 30, color = (0, 0.4, 1))
    ax1.set_xlim(0,1.05)
    ax1.set_ylim(0,1.05)

    # plot separator
    r = np.linspace(0,1,150)
    z = -w.item(0)/w.item(2) - w.item(1)/w.item(2)*r
    ax1.plot(r,z,color = color,linewidth = 2)

# main wrapper
def exp_vs_log_demo_hw():
    # load data
    X,y = load_data()

    # use compact notation and initialize
    temp = np.shape(X)
    temp = np.ones((temp[0],1))
    X_tilde = np.concatenate((temp,X),1)
    X_tilde = X_tilde.T

    alpha = 10**(-2)
    w0 = np.random.randn(3,1)

    # run gradient descent for h1
    w1 = gradient_descent_soft_cost(X_tilde,y,w0,alpha)

    # run gradient descent for h1
    w2 = gradient_descent_exp_cost(X_tilde,y,w0,alpha)

    # plot everything
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    plot_all(X,y,w1,'k',ax1)
    plot_all(X,y,w2,'m',ax1)
    plt.show()

exp_vs_log_demo_hw()
