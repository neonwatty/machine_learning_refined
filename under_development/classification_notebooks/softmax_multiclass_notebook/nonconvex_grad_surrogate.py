# nonconvex_grad_surrogate.py is a toy wrapper to illustrate the path
# taken by gradient descent.  The steps are evaluated
# at the objective, and then plotted.  For the first 4 iterations the
# linear surrogate used to transition from point to point is also plotted.
# The plotted points on the objective turn from green to red as the
# algorithm converges (or reaches a maximum iteration count, preset to 20).
#
# The (nonconvex) function here is
#
# # g(w) = exp(w)*cos(2pi*sin(pi*w))
#
# This file pairs with chapter 2 of the textbook "Machine Learning Refined" published by Cambridge University Press, 
# free for download at www.mlrefined.com

import numpy as np
import matplotlib.pyplot as plt
import time

# define the objective of our cost function
def obj(y):
    z = np.exp(y)*np.cos(2*np.pi*np.sin(np.pi*y))
    return z

# define the gradient of the cost function
def grad(y):
    z = np.exp(y)*np.cos(2*np.pi*np.sin(np.pi*y)) - 2*np.pi**2*np.exp(y)*np.sin(2*np.pi*np.sin(np.pi*y))*np.cos(np.pi*y)
    return z

# define the linear surrogate at each gradient descent step
def surrogate(y,x):
    z = obj(y) + grad(y)*(x - y)
    return z

# gradient descent function
def gradient_descent(w0,alpha):
    w = w0
    obj_path = []
    w_path = []
    w_path.append(w0)
    obj_path.append(np.exp(w)*np.cos(2*np.pi*np.sin(np.pi*w)))

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 50
    while np.linalg.norm(grad) > 10**(-5) and iter <= max_its:
        # take gradient step
        grad = np.exp(w)*np.cos(2*np.pi*np.sin(np.pi*w)) - 2*np.pi**2*np.exp(w)*np.sin(2*np.pi*np.sin(np.pi*w))*np.cos(np.pi*w)
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        obj_path.append(np.exp(w)*np.cos(2*np.pi*np.sin(np.pi*w)))
        iter+= 1

    # show final average gradient norm for sanity check
    s = grad**2
    s = 'The final average norm of the gradient = ' + str(float(s))
    print(s)
    return (w_path,obj_path)

# plots the cost function
def make_function():
    # plot the function
    global fig,ax1
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    s = np.linspace(0,1.1,2000)
    t =  np.exp(s)*np.cos(2*np.pi*np.sin(np.pi*s))
    ax1.plot(s,t,'-k',linewidth = 2)

    # pretty the figure up
    ax1.set_xlim(0,1.1)
    ax1.set_ylim(-5,5)
    ax1.set_xlabel('$w$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$g(w)$',fontsize=20,rotation = 0,labelpad = 20)

# plot each gradient descent step
def plot_steps(w_path,g_path):
    #colors for points
    s = np.linspace(1/len(g_path),1,len(g_path))
    s.shape = (len(s),1)
    colorspec = np.concatenate((s,np.flipud(s)),1)
    colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

    #plot initial point
    ax1.plot(w_path[0],g_path[0],'o',markersize = 9, color = colorspec[0,:], markerfacecolor = colorspec[0,:])
    plt.draw()
    # ax1.annotate('$w$'+str(0),(w_path[0],-5))
    t = np.linspace(-5,g_path[0],100)
    s = w_path[0]*np.ones((100))
    ax1.plot(s,t,'--k')
    plt.draw()
    time.sleep(2)

    #plot first surrogate and point traveled to
    s_range = .3
    s = np.linspace(w_path[0]-s_range,w_path[0]+s_range,10000)
    t = surrogate(w_path[0],s)
    h, = ax1.plot(s,t,'--m')
    r, = ax1.plot(w_path[1],surrogate(w_path[0],w_path[1]),'o',markersize = 6, c = 'k')
    plt.draw()

    for i in range(1,len(g_path)):
        if i < 4:
                time.sleep(3.5)

                #plot point
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                plt.draw()
                #plot surrogate
                if i < len(g_path) - 2:
                    time.sleep(1)
                    h.remove()
                    r.remove()
                    plt.draw()
                    s_range = .3
                    s = np.linspace(w_path[i]-s_range,w_path[i]+s_range,2000)
                    t = surrogate(w_path[i],s)
                    time.sleep(1)
                    h, = ax1.plot(s,t,'--m')
                    ind = np.argmin(abs(s - w_path[i + 1]))
                    r, = ax1.plot(s[ind],t[ind],'o', markersize = 6, c = 'k')
                    plt.draw()
        if i == 4:
            time.sleep(1.5)
            h.remove()
            r.remove()

        if i > 4: # just plot point so things don't get too cluttered
            time.sleep(0.01)
            ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
            plt.draw()
        if i == len(g_path) - 1:
            ax1.annotate('$w$'+str(i),(w_path[i],-5))
            t = np.linspace(-5,g_path[i],100)
            s = w_path[i]*np.ones((100))
            ax1.plot(s,t,'--k')

    plt.show(True)

# main
def main():
    alpha = 2*10**-3                             # step size
    make_function()                              # plot objective function
    pts = np.matrix(plt.ginput(1))
    x = pts[:,0]
    w0 = float(x[0])
    w_path,obj_path = gradient_descent(w0,alpha) # perform gradient descent
    plot_steps(w_path,obj_path)

main()
