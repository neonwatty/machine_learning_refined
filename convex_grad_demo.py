# convex_grad_surrogate.py is a toy wrapper to illustrate the path
# taken by gradient descent.  The steps are evaluated
# at the objective, and then plotted.  For the first 5 iterations the
# linear surrogate used to transition from point to point is also plotted.
# The plotted points on the objective turn from green to red as the
# algorithm converges (or reaches a maximum iteration count, preset to 50).
#
# The (convex) function here is
#
# g(w) = log(1 + exp(w^2))

# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from numpy import *
from matplotlib.pyplot import *
from pylab import *
import time

def obj(y):
    z = log(1 + exp(y**2))
    return z
def grad(y):
    z = (2*exp(y**2)*y)/(exp(y**2) + 1)
    return z
def surrogate(y,x):
    z = obj(y) + grad(y)*(x - y)
    return z

###### ML Algorithm functions ######
def gradient_descent(w0,alpha):
    w = w0
    obj_path = []
    w_path = []
    w_path.append(w0)
    obj_path.append(log(1 + exp(w**2)))

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 50
    while linalg.norm(grad) > 10**(-5) and iter <= max_its:
        # take gradient step
        grad = (2*exp(w**2)*w)/(exp(w**2) + 1)
        w = w - alpha*grad

        # update path containers
        w_path.append(w)
        obj_path.append(log(1 + exp(w**2)))
        iter+= 1
    # show final average gradient norm for sanity check
    s = grad**2
    s = 'The final average norm of the gradient = ' + str(float(s))
    print(s)

    # # for use in testing if algorithm minimizing/converging properly
    # obj_path = asarray(obj_path)
    # obj_path.shape = (iter,1)
    # plot(asarray(obj_path))
    # show()

    return (w_path,obj_path)

###### plotting functions #######
def make_function():
    # plot the function
    global fig,ax1
    fig = figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    s = linspace(-11,11,200)
    t = log(1 + exp(s**2))
    ax1.plot(s,t,'-k',linewidth = 2)

    # pretty the figure up
    ax1.set_xlim(-11,11)
    ax1.set_ylim(-20,120)
    ax1.set_xlabel('$w$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$g(w)$',fontsize=20,rotation = 0,labelpad = 20)
def plot_steps_with_surrogate(w_path,g_path):
    #colors for points
    s = linspace(1/len(g_path),1,len(g_path))
    s.shape = (len(s),1)
    colorspec = concatenate((s,flipud(s)),1)
    colorspec = concatenate((colorspec,zeros((len(s),1))),1)

    #plot initial point
    ax1.plot(w_path[0],g_path[0],'o',markersize = 9, color = colorspec[0,:], markerfacecolor = colorspec[0,:])
    draw()
    ax1.annotate('$w$'+str(0),(w_path[0],-20))
    t = linspace(-20,g_path[0],100)
    s = w_path[0]*ones((100))
    ax1.plot(s,t,'--k')
    draw()
    time.sleep(2)

    #plot first surrogate and point traveled to
    s_range = 5
    s = linspace(w_path[0]-s_range,w_path[0]+s_range,10000)
    t = surrogate(w_path[0],s)
    h, = ax1.plot(s,t,'--m')
    r, = ax1.plot(w_path[1],surrogate(w_path[0],w_path[1]),'o',markersize = 6, c = 'k')
    draw()

    for i in range(1,len(g_path)):
            if i < 5:
                time.sleep(1.5)

                #plot point
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()
                #plot surrogate
                if i < len(g_path) - 2:
                    time.sleep(1)
                    h.remove()
                    r.remove()
                    draw()
                    s_range = 5
                    s = linspace(w_path[i]-s_range,w_path[i]+s_range,2000)
                    t = surrogate(w_path[i],s)
                    time.sleep(1)
                    h, = ax1.plot(s,t,'--m')
                    ind = argmin(abs(s - w_path[i + 1]))
                    r, = ax1.plot(s[ind],t[ind],'o', markersize = 6, c = 'k')
                    draw()

            if i == 5:
                time.sleep(1.5)
                h.remove()
                r.remove()

            if i > 5: # just plot point so things don't get too cluttered
                time.sleep(0.05)
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()
            if i == len(g_path) - 1:
                ax1.annotate('$w$'+str(i),(w_path[i],-20))
                t = linspace(-20,g_path[i],100)
                s = w_path[i]*ones((100))
                ax1.plot(s,t,'--k')
    show(True)
def main():
    alpha = 0.15
    make_function()                             # plot objective function
    pts = matrix(ginput(1))
    x = pts[:,0]
    #scatter (x,obj(x), s=420, c='g', alpha=1)
    w0 = float(x[0])
    w_path,obj_path = gradient_descent(w0,alpha)    # perform gradient descent
    plot_steps_with_surrogate(w_path,obj_path)

main()
