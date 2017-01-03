# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt

# load the data
def load_data():
    data = np.matrix(np.genfromtxt('overlapping_2class.csv', delimiter=','))
    x = np.asarray(data[:,0:2])
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    return x,y

# run newton's method
def newtons_method(x,y):

    # begin newton's method loop
    k = 1
    max_its = 20

    while k <= max_its:
        # take Newton step
# ----> w =

        # update counter
        k+=1
    return w

# calculate the objective value for a given input weight w
def calculate_obj(X,y,w):
    obj = np.log(1 + my_exp(-y*np.dot(X.T,w)))
    obj = obj.sum()
    return obj

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u

# plotting function
def plot_fit(x,y,w):

    # initialize figure, plot data, and dress up panels with axes labels etc.
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$x_1$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 20)
    s = np.argwhere(y == 1)
    s = s[:,0]
    plt.scatter(x[s,0],x[s,1], s = 30,color = (1, 0, 0.4))
    s = np.argwhere(y == -1)
    s = s[:,0]
    plt.scatter(x[s,0],x[s,1],s = 30, color = (0, 0.4, 1))
    ax1.set_xlim(min(x[:,0])-0.1, max(x[:,0])+0.1)
    ax1.set_ylim(min(x[:,1])-0.1,max(x[:,1])+0.1)

    # plot separator
    r = np.linspace(0,1,150)
    z = -w.item(0)/w.item(2) - w.item(1)/w.item(2)*r
    ax1.plot(r,z,'-k',linewidth = 2)
    plt.show()

# main
def main():
    # load data
    x,y = load_data()

    # run newtons method to minimize logistic regression or softmax cost
    w = newtons_method(x,y)

    # plot everything - including data and separator
    plot_fit(x,y,w)
    plt.show()

main()
