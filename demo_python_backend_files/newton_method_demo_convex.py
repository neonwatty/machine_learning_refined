import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

###### functions defining various aspects of the cost function #######
def obj(y):
    z = y**4 + y**2 + 10*y
    return z
def grad(y):
    z = 4*(y**3) + 2*y + 10 
    return z
def hess(y):
    z = 12*y**2 + 2
    return z
def surrogate(y,x):
    z = obj(y) + grad(y)*(x - y) + 0.5*hess(y)*(x - y)*(x - y)
    return z

###### plotting functions #######

# plot the underlying function
def make_function(ax):
    s = np.linspace(-3,2,200)
    t = obj(s)
    ax.plot(s,t,'-k',linewidth = 2)

# this plots the surrogate function at each step of Newton's method
def plot_steps_with_surrogate(w_history):
    # generate path of cost function corresponding to w_history
    g_history = []
    for i in range(len(w_history)):
        w = w_history[i]
        g = obj(w)
        g_history.append(g)
        
    # make original figure
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    make_function(ax1)
    
    # pretty the figure up
    ax1.set_xlim([-3,2])
    ax1.set_ylim([-30,60])
    ax1.set_xlabel('$w$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$g(w)$',fontsize=20,rotation = 0,labelpad = 20)
        
    # colors for points - change from green to red as we get closer to max_its number
    s = np.linspace(1/len(g_history),1,len(g_history))
    s.shape = (len(s),1)
    colorspec = np.concatenate((s,np.flipud(s)),1)
    colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

    # plot initial point
    ax1.plot(w_history[0],g_history[0],'o',markersize = 9, color = colorspec[0,:], markerfacecolor = colorspec[0,:])
    
    t = np.linspace(-30,g_history[0],100)
    s = w_history[0]*np.ones((100))
    ax1.plot(s,t,'--k')

    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    time.sleep(2)

    # plot first surrogate and point traveled to
    s_range = 2
    s = np.linspace(w_history[0]-s_range,w_history[0]+s_range,10000)
    t = surrogate(w_history[0],s)
    h, = ax1.plot(s,t,'--m')
    time.sleep(0.5)
    ind = np.argmin(t)
    x_mark, = ax1.plot(s[ind],t[ind],'*',markersize = 11, c = 'k')
    time.sleep(0.5)
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    
    for i in range(1,len(g_history)):
        if i < 3:
            time.sleep(1.5)

            # plot point
            ax1.plot(w_history[i],g_history[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
            display.clear_output(wait=True)
            display.display(plt.gcf()) 

            time.sleep(1)
            h.remove()
            x_mark.remove()
            display.clear_output(wait=True)
            display.display(plt.gcf()) 

            s_range = 2
            s = np.linspace(w_history[i]-s_range,w_history[i]+s_range,10000)
            t = surrogate(w_history[i],s)
            h, = ax1.plot(s,t,'--m')
            time.sleep(0.5)
            ind = np.argmin(t)
            x_mark, = ax1.plot(s[ind],t[ind],'*',markersize = 11, c = 'k')
            time.sleep(0.5)
            display.clear_output(wait=True)
            display.display(plt.gcf()) 

        if i > 5: # just plot point so things don't get too cluttered
            time.sleep(0.05)
            ax1.plot(w_history[i],g_history[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])
            display.clear_output(wait=True)
            display.display(plt.gcf()) 

        if i == len(g_history) - 1:
            h.remove()
            x_mark.remove()
            time.sleep(1.5)
            
            ax1.plot(w_history[i],g_history[i],'o',markersize = 9, color = colorspec[i-1,:], markerfacecolor = colorspec[i-1,:])

            t = np.linspace(-30,g_history[i],100)
            s = w_history[i]*np.ones((100))
            ax1.plot(s,t,'--k')

            display.clear_output(wait=True)
            display.display(plt.gcf()) 