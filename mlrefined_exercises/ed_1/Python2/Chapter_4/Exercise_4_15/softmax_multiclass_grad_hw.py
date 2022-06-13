# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the data
def load_data():
    # load data
    data = np.matrix(np.genfromtxt('4class_data.csv', delimiter=','))
    x = np.asarray(data[:,0:2])
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    return x,y

###### ML Algorithm functions ######
# learn all C separators running gradient descent
def gradient_descent(x,y,alpha):
    # formulate full input data matrix X
    temp = np.shape(x)
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    num_classes = len(np.unique(y))
    num_pts = len(y)
    W = np.random.randn(3,num_classes)

    # record objective value at each iteration to check that algorithm is working properly
    obj_path = []
    obj = calculate_obj(X,y,W)
    obj_path.append(obj)

    # gradient descent loop
    iter = 1
    max_its = 1000
    while iter < max_its:

        # calculate gradient
# --->  grad = 

        # full gradient completely calculated - take gradient step
        W = W - alpha*grad
        iter+= 1

        # update path containers - used to check that algorithm is working
        obj = calculate_obj(X,y,W)
        obj_path.append(obj)

    # plot objective value at each iteration to make sure everything works properly
    obj_path = np.asarray(obj_path)
    obj_path.shape = (max_its,1)
    # plt.plot(np.asarray(obj_path))

    return W

# calculate the objective value for a given input weight w
def calculate_obj(X,y,W):
    # loop for cost function
    cost = 0
    for p in range(0,len(y)):
        s = int(y[p])-1
        p_temp = 0
        for j in range(0,len(np.unique(y))-1):
            p_temp += np.exp(np.dot(X[:,p].T,(W[:,j] - W[:,s])))
        p_temp = np.log(p_temp)

        # update cost
        cost+=p_temp
    return cost

# plot data, separators, and fused fule
def plot_all(x,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.
    num_classes = np.size(np.unique(y))
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,3,facecolor = 'white')
    for a in range(0,3):
        for i in range(0,num_classes):
            s = np.argwhere(y == i+1)
            s = s[:,0]
            axs[a].scatter(x[s,0],x[s,1], s = 30,color = color_opts[i,:])

        # dress panel correctly
        axs[a].set_xlabel('$x_1$',fontsize=20,labelpad = 20)
        axs[a].set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 20)
        axs[a].set_xlim(0,1)
        axs[a].set_ylim(0,1)
        axs[a].set(aspect = 'equal')

    r = np.linspace(0,1,150)
    for i in range(0,num_classes):
        z = -W[0,i]/W[2,i] - W[1,i]/W[2,i]*r
        axs[1].plot(r,z,'-k',linewidth = 2,color = color_opts[i,:])

    # fuse individual subproblem separators into one joint rule
    r = np.linspace(0,1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((np.ones((np.size(s),1)),s,t),1)
    f = np.dot(W.T,h.T)
    z = np.argmax(f,0)
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    for i in range(0,num_classes + 1):
        axs[2].contour(s,t,z,num_classes-1,colors = 'k',linewidths = 2.25)

# main wrapper
def softmax_multiclass_grad_hw():
    # load data
    x,y = load_data()

    # perform gradient descent on softmax multiclass
    alpha = 10**(-2)    # step length, tune to your heat's desire!
    W = gradient_descent(x,y,alpha)           # learn all C vs notC separators
    plot_all(x,y,W)
    plt.show()

softmax_multiclass_grad_hw()
