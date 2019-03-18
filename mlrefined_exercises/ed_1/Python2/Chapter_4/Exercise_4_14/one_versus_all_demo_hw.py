# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt

# load the data
def load_data():
    # load data
    data = np.matrix(np.genfromtxt('four_class_data.csv', delimiter=','))
    x = np.asarray(data[:,0:2])
    temp = np.shape(x)
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    return X,y

###### ML Algorithm functions ######
# learn all C separators
def learn_separators(X,y):
    W = []
    num_classes = np.size(np.unique(y))
    for i in range(0,num_classes):
        # prepare temporary C vs notC probem labels
        y_temp = np.copy(y)
        ind = np.argwhere(y_temp == (i+1))
        ind = ind[:,0]
        ind2 = np.argwhere(y_temp != (i+1))
        ind2 = ind2[:,0]
        y_temp[ind] = 1
        y_temp[ind2] = -1
        # run descent algorithm to classify C vs notC problem
        w = newtons_method(np.random.randn(3,1),X,y_temp)
        W.append(w)
    W = np.asarray(W)
    W.shape = (num_classes,3)
    W = W.T
    return W

# run newton's method
def newtons_method(w0,X,y):

# ---> YOU MUST COMPLETE THIS MODULE.    

    return w

# calculate the objective value for a given input weight w
def calculate_obj(w,X,y):
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
def plot_data_and_subproblem_separators(X,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.
    num_classes = np.size(np.unique(y))
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,num_classes + 1,facecolor = 'white')

    r = np.linspace(0,1,150)
    for a in range(0,num_classes):
        # color current class
        axs[a].scatter(X[1,],X[2,], s = 30,color = '0.75')
        s = np.argwhere(y == a+1)
        s = s[:,0]
        axs[a].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])
        axs[num_classes].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])

        # draw subproblem separator
        z = -W[0,a]/W[2,a] - W[1,a]/W[2,a]*r
        axs[a].plot(r,z,'-k',linewidth = 2,color = color_opts[a,:])

        # dress panel correctly
        axs[a].set_xlim(0,1)
        axs[a].set_ylim(0,1)
        axs[a].set(aspect = 'equal')
    axs[num_classes].set(aspect = 'equal')

    return axs

# fuse individual subproblem separators into one joint rule
def plot_joint_separator(W,axs,num_classes):
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
        axs[num_classes].contour(s,t,z,num_classes-1,colors = 'k',linewidths = 2.25)

def one_versus_all_demo_hw():
    # load the data
    X,y = load_data()

    # learn all C vs notC separators
    W = learn_separators(X,y)

    # plot data and each subproblem 2-class separator
    axs = plot_data_and_subproblem_separators(X,y,W)

    # plot fused separator
    plot_joint_separator(W,axs,np.size(np.unique(y)))

    plt.show()

one_versus_all_demo_hw()
