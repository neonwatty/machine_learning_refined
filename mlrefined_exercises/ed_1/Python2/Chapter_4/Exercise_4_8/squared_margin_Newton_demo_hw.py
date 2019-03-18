# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import matplotlib.pyplot as plt
import numpy as np

# load the data
def load_data():
    # load data
    global X,y,ax1

    data = np.matrix(np.genfromtxt('overlapping_2class.csv', delimiter=','))
    x = np.asarray(data[:,0:2])
    temp = np.shape(x)
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)

    # initialize figure, plot data, and dress up panels with axes labels etc.
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$x$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 20)
    s = np.argwhere(y == 1)
    s = s[:,0]
    plt.scatter(x[s,0],x[s,1], s = 30,color = (1, 0, 0.4))
    s = np.argwhere(y == -1)
    s = s[:,0]
    plt.scatter(x[s,0],x[s,1],s = 30, color = (0, 0.4, 1))
    ax1.set_xlim(min(x[:,0])-0.1, max(x[:,0])+0.1)
    ax1.set_ylim(min(x[:,1])-0.1,max(x[:,1])+0.1)

###### ML Algorithm functions ######
# run newton's method
def newtons_method(w0):
    obj_path = []
    obj = calculate_obj(w0)
    obj_path.append(obj)
    w = w0
    grad = 1
    iter = 1
    max_its = 100
    while np.linalg.norm(grad) > 10**(-5) and iter <= max_its:

        # take Newton step
# --->  w =

        # update path containers
        obj = calculate_obj(w)
        obj_path.append(obj)
        iter+= 1

    # for use in testing if algorithm minimizing/converging properly
    obj_path = np.asarray(obj_path)
    obj_path.shape = (iter,1)
    #plot(asarray(obj_path))

    return w


# calculate the objective value for a given input weight w
def calculate_obj(w):
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

###### plotting functions #######
def plot_fit(w):
    r = np.linspace(0,1,150)
    z = -w.item(0)/w.item(2) - w.item(1)/w.item(2)*r
    ax1.plot(r,z,'k-',linewidth = 2)

def squared_margin_Newton_demo_hw():
    load_data()              # load the data

    ### run newtons method with first initial point
    w0 = np.random.randn(3,1)
    w = newtons_method(w0)
    plot_fit(w)
    plt.show()

squared_margin_Newton_demo_hw()
