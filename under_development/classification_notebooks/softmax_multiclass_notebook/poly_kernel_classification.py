from pylab import *
from mpl_toolkits.mplot3d import Axes3D

###### loading and plotting functions ######
# loads data
def load_data():
    # load data
    data = matrix(genfromtxt('2_eggs.csv', delimiter=','))
    x = asarray(data[:,0:-1])
    y = asarray(data[:,-1])
    return (x,y)

# plots data
def plot_data(x,y):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    global ax1,ax2

    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection = '3d')

    # note to future self - you cannot plot a 3d surface and scatter on the same panel, so you need to use this work around
    a = argwhere(y == 1)
    a = a[:,0]
    ax1.scatter(x[a,0],x[a,1],color = (1, 0, 0.4),s = 30)
    x_temp = asarray(x[a,:])
    y_temp = asarray(y[a])
    ax2.plot(x_temp[:,0],x_temp[:,1],y_temp[:,0],color = (1, 0, 0.4),marker = 'o',linewidth = 5,linestyle = 'none')

    a = argwhere(y == -1)
    a = a[:,0]
    ax1.scatter(x[a,0],x[a,1],color = (0, 0.4, 1),s = 30)

    # note to future self - you cannot plot a 3d surface and scatter on the same panel, so you need to use this work around
    x_temp = asarray(x[a,:])
    y_temp = asarray(y[a])
    ax2.plot(x_temp[:,0],x_temp[:,1],y_temp[:,0],color =(0, 0.4, 1),marker = 'o',linewidth = 5, linestyle = 'none')

    # make plots nice
    ax1.set_xlim(min(x[:,0])-0.05, max(x[:,0])+0.05)
    ax1.set_ylim(min(x[:,1])-0.05,max(x[:,1])+0.05)
    ax1.set_xlabel('$x$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 20)
    ax2.set_xlim(min(x[:,0])-0.05, max(x[:,0])+0.05)
    ax2.set_ylim(min(x[:,1])-0.05,max(x[:,1])+0.05)
    ax2.xaxis.set_rotate_label(False)
    ax2.yaxis.set_rotate_label(False)
    ax2.zaxis.set_rotate_label(False)
    ax2.get_xaxis().set_ticks([0,1])
    ax2.get_yaxis().set_ticks([0,1])
    ax2.set_xlabel('$w_0$   ',fontsize=20,rotation = 0,linespacing = 10)
    ax2.set_ylabel('$w_1$',fontsize=20,rotation = 0,labelpad = 50)
    ax2.set_zlabel('   $g(\mathbf{w})$',fontsize=20,rotation = 0,labelpad = 20)

# plots objective function decrease, used to make sure algorithm runs correctly
def plot_obj(g_path):
    # for use in testing if algorithm minimizing/converging properly
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    g_path = asarray(g_path)
    g_path.shape = (size(g_path),1)
    ax1.plot(asarray(g_path))
    ax1.set_xlabel('iteration',fontsize=20,labelpad = 20)
    ax1.set_ylabel('objective value',fontsize=20,rotation = 90,labelpad = 20)
    ax1.set_title('objective value')
    show()

# plots contour and surface fit
def plot_fit(x,w,deg):
    s = linspace(0,1,200)
    s1, s2 = meshgrid(s, s)
    s1.shape = (size(s1),1)
    s2.shape = (size(s2),1)
    f = zeros((size(s1),1))

    for i in range(0,size(s1)):
        p = array([s1[i],s2[i]])
        p.shape = (2,1)
        d = (dot(x,p) + 1)**deg
        m = vstack((1,d))
        m = dot(m.T,w)
        f[i] = sign(2*sigmoid(m) -1)

    # reshape for plotting contour and surface
    s1.shape = (size(s),size(s))
    s2.shape = (size(s),size(s))
    f.shape = (size(s),size(s))

    # compute gradient to find boundary of top and bottom step
    ax1.contour(s1,s2,f,(0,),colors ='k')
    e,r = gradient(f)
    mag = e**2 + r**2
    f[mag !=0] = NaN
    ax2.plot_surface(s1,s2,f,rstride=4, cstride=4,cmap = 'summer',alpha = 0.5,linewidth = 0)

###### ML Algorithm functions ######
# run newton's method
def softmax_newtons_method(H,y,w,max_its):
    g_path = []     # records evaluation at each step, useful in making sure algorithm works well
    lam = 10**-5    # small regularizer used due to nonsingularity of kernel matrix

    # pre-make arrays for computation
    temp = ones((size(w) - 1,1))
    H2 = concatenate((temp,H),1)
    A = H2*y
    s = shape(y)[0]
    l = ones((s,1))
    for iter in range(0,max_its):
        # compute gradient
        temp = sigmoid(-dot(A,w))
        grad = - dot(A.T,temp)
        grad[1:] += 2*lam*dot(H,w[1:])

        # compute Hessian
        g = temp*(l - temp)
        temp = H2*g
        temp = temp.T
        hess = dot(temp,H2)
        hess[1:,1:] += 2*lam*H

        # take Newton step = solve Newton system
        temp = dot(hess,w) - grad
        w = linalg.lstsq(hess, temp)[0]

        # update path containers - turn on to output each step evaluated by objective
        if iter > 0:
            obj = calculate_obj(H2,y,w)
            g_path.append(obj)

    return w

# calculate the objective value for a given input weight w
def calculate_obj(H2,y,w):
    obj = log(1 + my_exp(-y*dot(H2,w)))
    obj = obj.sum()
    return obj

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = argwhere(u > 100)
    t = argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = exp(u)
    u[t] = 1
    return u

# sigmoid function, uses my_exp
def sigmoid(t):
    z = 1/(1 + my_exp(-t))
    return z

# makes polynomial kernel
def poly_kernel(A,deg):
    n = shape(A)[0]
    H = eye(n)
    for i in range(0,n):
        for j in range(0,n):
            temp = (1 + dot(A[i,:],A[j,:].T))**deg
            H[i,j] = temp
            H[j,i] = temp

    return H

def main():
    deg = 5
    x,y = load_data()              # load the data
    H = poly_kernel(x,deg)         # make poly kernel of data

    ### run newtons method with first initial point
    w = zeros((shape(H)[0] + 1,1))
    max_its = 20
    w = softmax_newtons_method(H,y,w,max_its)
    # plot_obj(g_path)
    plot_data(x,y)
    plot_fit(x,w,deg)
    show()

main()