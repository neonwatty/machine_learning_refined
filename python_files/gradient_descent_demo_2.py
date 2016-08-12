# convex_grad_surrogate.py is a toy wrapper to illustrate the path
# taken by gradient descent.  The steps are evaluated
# at the objective, and then plotted.  For the first 4 iterations the
# linear surrogate used to transition from point to point is also plotted.
# The plotted points on the objective turn from green to red as the
# algorithm converges (or reaches a maximum iteration count, preset to 50).
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
from IPython import display

# calculate cost function at input w
def calculate_cost_value(w):
    g = np.exp(w)*np.cos(2*np.pi*np.sin(np.pi*w))
    return g

# calculate the gradient of the cost function at an input w
def calculate_gradient(w):
    grad = np.exp(w)*np.cos(2*np.pi*np.sin(np.pi*w)) - 2*np.pi**2*np.exp(w)*np.sin(2*np.pi*np.sin(np.pi*w))*np.cos(np.pi*w)
    return grad

# define the linear surrogate generated at each gradient step
def surrogate(y,x):
    g = calculate_cost_value(y)
    grad = calculate_gradient(y)
    h = g + grad*(x - y)
    return h

# the gradient descent function
def gradient_descent(w,alpha,max_its):
    w_history = []
    w_history.append(w)

    # start gradient descent loop
    iter = 1
    for k in range(0,max_its):
        # calculate gradient (a single derivative in this case)
        grad = calculate_gradient(w)
        
        # take gradient descent step
        w = w - alpha*grad

        # store weights at this iteration
        w_history.append(w)

    return w_history

# show cost function to minimize
def make_function(ax):
    s = np.linspace(0,1.1,200)
    t =  np.exp(s)*np.cos(2*np.pi*np.sin(np.pi*s))
    ax.plot(s,t,'-k',linewidth = 2)

    # pretty the figure up
    ax.set_xlim(0,1.1)
    ax.set_ylim(-5,5)
    ax.set_xlabel('$w$',fontsize=20,labelpad = 20)
    ax.set_ylabel('$g(w)$',fontsize=20,rotation = 0,labelpad = 20)

# plot each step of gradient descent
def plot_steps_with_surrogate(w_path):
    # make figure to update
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(111)
    
    # make cost function path based on gradient descent steps (in w_path)
    g_path = []
    for i in range(0,len(w_path)):
        w = w_path[i]
        g_path.append(calculate_cost_value(w))
        
    # plot costs function
    make_function(ax1)  
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    
    #colors for points
    s = np.linspace(1/len(g_path),1,len(g_path))
    s.shape = (len(s),1)
    colorspec = np.concatenate((s,np.flipud(s)),1)
    colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

    # plot initial point
    ax1.plot(w_path[0],g_path[0],'o',markersize = 8, color = colorspec[0,:], markerfacecolor = colorspec[0,:])
    display.clear_output(wait=True)
    display.display(plt.gcf())   
    
    # plot a tracer on this first point just for visualization purposes
    t = np.linspace(-20,g_path[0],100)
    s = w_path[0]*np.ones((100))
    ax1.plot(s,t,'--k')
    display.clear_output(wait=True)
    display.display(plt.gcf())     
    time.sleep(1)

    # plot first surrogate and point traveled to on that surrogate
    s_range = 0.2  # range over which to show the linear surrogate
    s = np.linspace(w_path[0]-s_range,w_path[0]+s_range,1000)
    t = surrogate(w_path[0],s)
    h, = ax1.plot(s,t,'m',linewidth = 2)
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    time.sleep(0.5)
    r, = ax1.plot(w_path[1],surrogate(w_path[0],w_path[1]),marker = '*',markersize = 11, c = 'k')
    display.clear_output(wait=True)
    display.display(plt.gcf()) 

    # loop over the remaining iterations, showing
    # - the linear surrogate at the first few steps
    # - color changing from green (start) to red (end) of gradient descent run
    for i in range(1,len(g_path)):
            # with the first few points plot the surrogates as well for illustration
            if i < 4:
                time.sleep(1.5)

                # plot cost function evaluated at next gradient descent step
                ax1.plot(w_path[i],g_path[i],'o',markersize = 8, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                display.clear_output(wait=True)
                display.display(plt.gcf()) 
                
                # plot linear surrogate and point traveled too
                if i < 3:
                    # remove previously drawn linear surrogate, point on it, etc.,
                    h.remove()
                    r.remove()
                    time.sleep(1)
                    display.clear_output(wait=True)
                    display.display(plt.gcf())  
                    
                    # generate linear surrogate and plot
                    s_range = 5
                    s = np.linspace(w_path[i]-s_range,w_path[i]+s_range,2000)
                    t = surrogate(w_path[i],s)
                    time.sleep(1)
                    h, = ax1.plot(s,t,'m',linewidth = 2)
                    display.clear_output(wait=True)
                    display.display(plt.gcf()) 
                    time.sleep(0.5)

                    # generate point on linear surrogate we travel too
                    ind = np.argmin(abs(s - w_path[i + 1]))
                    r, = ax1.plot(s[ind],t[ind],marker = '*', markersize = 11, c = 'k')
                    display.clear_output(wait=True)
                    display.display(plt.gcf()) 
            
            # remove linear surrogate, point, etc.,
            if i == 4:
                time.sleep(0.5)
                h.remove()
                r.remove()
                display.clear_output(wait=True)
                display.display(plt.gcf())     
                time.sleep(1)

            # for later iterations just plot point so things don't get too visually cluttered
            if i > 4: 
                time.sleep(0.05)
                ax1.plot(w_path[i],g_path[i],'o',markersize = 10, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
                display.clear_output(wait=True)
                display.display(plt.gcf()) 

            # color the final point red just for visualization purposes
            if i == len(g_path) - 1:
                t = np.linspace(-20,g_path[i],100)
                s = w_path[i]*np.ones((100))
                ax1.plot(s,t,'--k')
                display.clear_output(wait=True)
                display.display(plt.gcf()) 

# main function
def run(w0,alpha,max_its):
    # run gradient descent with chosen initial w0 and step length alpha
    w_history = gradient_descent(w0,alpha,max_its)    
    
    # plot all of the gradient descent steps evaluated at the cost function
    plot_steps_with_surrogate(w_history)
    display.clear_output(wait=True)


