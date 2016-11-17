# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from mpl_toolkits.mplot3d import Axes3D
from pylab import *

# load the data
def load_data():
    # load data
    global X,y,ax1,ax2

    data = matrix(genfromtxt('bacteria_data.csv', delimiter=','))
    x = asarray(data[:,0])
    temp = ones((size(x),1))
    X = concatenate((temp,x),1)
    y = asarray(data[:,1])
    y = y/y.max()

    # initialize figure, plot data, and dress up panels with axes labels etc.,
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('$x$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 20)
    plot(x,y,'ko')
    ax1.set_xlim(min(x[:,0])-0.5, max(x[:,0])+0.5)
    ax1.set_ylim(min(y)-0.1,max(y)+0.1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.xaxis.set_rotate_label(False)
    ax2.yaxis.set_rotate_label(False)
    ax2.zaxis.set_rotate_label(False)
    ax2.get_xaxis().set_ticks([-3,-1,1,3])
    ax2.get_yaxis().set_ticks([-3,-1,1,3])
    ax2.set_xlabel('$w_0$   ',fontsize=20,rotation = 0,linespacing = 10)
    ax2.set_ylabel('$w_1$',fontsize=20,rotation = 0,labelpad = 50)
    ax2.set_zlabel('   $g(\mathbf{w})$',fontsize=20,rotation = 0,labelpad = 20)

###### ML Algorithm functions ######
# run gradient descent
def gradient_descent(w0):
    w_path = []         # container for weights learned at each iteration
    obj_path = []       # container for associated objective values at each iteration
    w_path.append(w0)
    obj = calculate_obj(w0)
    obj_path.append(obj)
    grad_path = []
    w = w0

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 5000
    alpha = 10**(-2)
    while linalg.norm(grad) > 10**(-5) and iter <= max_its:
        # compute gradient
# --->  grad =

        # take gradient step
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        obj = calculate_obj(w)
        obj_path.append(obj)
        iter+= 1

    # reshape containers for use in plotting in 3d
    w_path = asarray(w_path)
    w_path.shape = (iter,2)
    obj_path = asarray(obj_path)
    obj_path.shape = (iter,1)

    return (w_path,obj_path)
    ## for use in testing if gradient vanishes
    # grad_path = asarray(grad_path)
    # grad_path.shape = (iter,1)
    # plot(asarray(grad_path))
    # show()

# calculate the objective value for a given input weight w
def calculate_obj(w):
    temp = 1/(1 + my_exp(-dot(X,w))) - y
    temp = dot(temp.T,temp)
    return temp

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = argwhere(u > 100)
    t = argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = exp(u)
    u[t] = 1
    return u

###### plotting functions #######
# make 3d surface plot
def plot_logistic_surface():
    r = linspace(-3,3,150)
    s,t = meshgrid(r, r)
    s = reshape(s,(size(s),1))
    t = reshape(t,(size(t),1))
    h = concatenate((s,t),1)

    # build 3d surface
    surf = zeros((size(s),1))
    max_its = size(y)
    for i in range(0,max_its):
        surf = surf + add_layer(X[i,:],y[i],h)


    s = reshape(s,(sqrt(size(s)),sqrt(size(s))))
    t = reshape(t,(sqrt(size(t)),sqrt(size(t))))
    surf = reshape(surf,(sqrt(size(surf)),sqrt(size(surf))))

    # plot 3d surface
    ax2.plot_surface(s,t,surf,cmap = 'jet')
    ax2.azim = 175
    ax2.elev = 20

# used by plot_logistic_surface to make objective surface of logistic regression cost function
def add_layer(a,b,c):
    a.shape = (2,1)
    b.shape = (1,1)
    z = my_exp(-dot(c,a))
    z = 1/(1 + z) - b
    z = z**2
    return z

# plot fit to data and corresponding gradient descent path onto the logistic regression objective surface
def show_fit_and_paths(w_path,obj_path,col):
    # plot solution of gradient descent fit to original data
    s = linspace(0,X.max(),100)
    t = 1/(1 + my_exp(-(w_path[-1,0] + w_path[-1,1]*s)))
    ax1.plot(s,t,color = col)

    # plot grad descent path onto surface
    ax2.plot(w_path[:,0],w_path[:,1],obj_path[:,0],color = col,linewidth = 5)   # add a little to output path so its visible on top of the surface plot

def main():

    load_data() # load in data

    plot_logistic_surface()  # plot objective surface

    ### run gradient descent with first initial point
    w0 = array([0,2])
    w0.shape = (2,1)
    w_path, obj_path = gradient_descent(w0)

    # plot fit to data and path on objective surface
    show_fit_and_paths(w_path, obj_path,'m')

    ### run gradient descent with first initial point
    w0 = array([0,-2])
    w0.shape = (2,1)
    w_path, obj_path = gradient_descent(w0)

    # plot fit to data and path on objective surface
    show_fit_and_paths(w_path, obj_path,'c')
    show()

main()
