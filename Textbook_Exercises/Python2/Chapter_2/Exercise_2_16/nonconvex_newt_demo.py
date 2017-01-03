# nonconvex_newt_demo.py is a toy wrapper to illustrate the path
# taken by Hessian descent (or Newton's method).  The steps are evaluated
# at the objective, and then plotted.  For the first 5 iterations the
# quadratic surrogate used to transition from point to point is also plotted.
# The plotted points on the objective turn from green to red as the
# algorithm converges (or reaches a maximum iteration count, preset to 50).

# The (nonconvex) function here is
#
# g(w) = exp(w)*cos(2pi*sin(pi*w))

# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from numpy import *
from matplotlib.pyplot import *
from pylab import *
import time

def obj(y):
    z = exp(y)*cos(2*pi*sin(pi*y))
    return z
def grad(y):
    z = exp(y)*cos(2*pi*sin(pi*y)) - 2*(pi**2)*exp(y)*sin(2*pi*sin(pi*y))*cos(pi*y)
    return z
def hess(y):
    z = exp(y)*(2*(pi**3)*sin(pi*y)*sin(2*pi*sin(pi*y)) - (4*pi**4)*(cos(pi*y)**2)*cos(2*pi*sin(pi*y)) + cos(2*pi*sin(pi*y)) - (4*pi**2)*sin(2*pi*sin(pi*y))*cos(pi*y))
    return z
def surrogate(y,x):
    a = hess(y)
    z = obj(y) + grad(y)*(x - y) + 0.5*hess(y)*(x - y)*(x - y)
    return z,a


###### ML Algorithm functions ######
def hessian_descent(w0):

    #initializations
    grad_stop = 10**-3
    max_its = 50
    iter = 1
    grad_eval = 1
    g_path = []
    w_path = []
    w_path.append(w0)
    g_path.append(obj(w0))
    w = w0
    #main loop
    while linalg.norm(grad_eval) > grad_stop and iter <= max_its:
        #take gradient step
        grad_eval = grad(w)
        hess_eval = hess(w)
        w = w - grad_eval/hess_eval

        #update containers
        w_path.append(w)
        g_path.append(exp(w)*cos(2*pi*sin(pi*w)))

        #update stopers
        iter+= 1
    return w,w_path, g_path

###### plotting functions #######
def make_function():
    # plot the function
    global fig,ax1
    fig = figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    s = linspace(0,1.1,2000)
    t =  exp(s)*cos(2*pi*sin(pi*s))
    ax1.plot(s,t,'-k',linewidth = 2)

    # pretty the figure up
    ax1.set_xlim(0,1.1)
    ax1.set_ylim(-5,5)
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
    ax1.annotate('$w$'+str(0),(w_path[0],-5))
    t = linspace(-5,g_path[0],100)
    s = w_path[0]*ones((100))
    ax1.plot(s,t,'--k')
    draw()
    time.sleep(2)

    #plot first surrogate and point traveled to
    s_range = 5
    s = linspace(w_path[0]-s_range,w_path[0]+s_range,10000)
    t,a = surrogate(w_path[0],s)
    h, = ax1.plot(s,t,'--m')
    if a>0:
        ind = argmin(t)
    if a<0:
        ind = argmax(t)
    x_mark, = ax1.plot(s[ind],t[ind],'ko',markersize = 6)
    draw()

    for i in range(1,len(g_path)):
            if i < 5:
                time.sleep(1.5)

                #plot point
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()
                time.sleep(1)
                h.remove()
                x_mark.remove()
                draw()
                s_range = 5
                s = linspace(w_path[i]-s_range,w_path[i]+s_range,10000)
                t,a = surrogate(w_path[i],s)
                time.sleep(1)
                h, = ax1.plot(s,t,'--m')
                if a>0:
                    ind = argmin(t)
                if a<0:
                    ind = argmax(t)
                x_mark, = ax1.plot(s[ind],t[ind],'ko',markersize = 6)
                draw()


            if i > 5: # just plot point so things don't get too cluttered
                time.sleep(0.05)
                ax1.plot(w_path[i],g_path[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                draw()
            if i == len(g_path) - 1:
                h.remove()
                x_mark.remove()
                time.sleep(1.5)
                ax1.annotate('$w$'+str(i),(w_path[i],-5))
                t = linspace(-5,g_path[i],100)
                s = w_path[i]*ones((100))
                ax1.plot(s,t,'--k')
    show(True)

def main():
    make_function()                             # plot objective function
    pts = matrix(ginput(1))
    x = pts[:,0]
    w0 = float(x[0])
    w,w_path,g_path = hessian_descent(w0)    # perform gradient descent
    plot_steps_with_surrogate(w_path,g_path)

main()
