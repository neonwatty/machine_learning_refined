# This file pairs with chapter 3 of the textbook "Machine Learning Refined" published by Cambridge University Press, 
# free for download at www.mlrefined.com
import csv
import math
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import display

def load_data():
    # grab data and labels
    a = 0
    x = []
    y = []
    count = 0
    with open('datasets/student_debt.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if count == 0:
                a = row
            else:
                x.append(float(row[0]))
                y.append(float(row[1]))
            count+=1
    x = np.asarray(x)
    x.shape = (len(x),1)
    x = x - np.mean(x)     # normalize input
    y = np.asarray(y)
    y.shape = (len(y),1)

    return x,y
########## calculate cost function values ##########
# calculate cost function at an input step
def calculate_cost_value(X,y,w):
    # define limits
    P = len(y)
    
    # loop for cost function
    cost = 0
    for p in range(0,P):
        y_p = y[p]
        x_p = X[:,p]
        cost += (np.dot(x_p.T,w) - y_p)**2
        
    return cost  

# plot data
def plot_pts(x,y,ax):
    # plot points
    ax.plot(x, y,'ko',label = 'student debt')
    
    # dress up graph
    ax.set_xlabel('year',fontsize = 15)
    ax.set_ylabel('total debt',fontsize = 15)
    ax.set_title('total U.S. student debt (in trillions of dollars)',fontsize = 17)
    ax.set_xlim(np.min(x)-1, np.max(x)+2)
    ax.set_ylim(0.9*np.min(y),1.4*np.max(y))
    s = np.linspace(np.min(x),np.max(x) + 4*(np.max(x) - np.min(x))/float(7), 9)
    ax.set_xticks(s)
    ax.set_xticklabels([2004,2006,2008,2010,2012,2014,2016,2018,2020])
    
# plot fit
def plot_fit(x,y,w,ax):
    # generate area over which to make line
    s = np.linspace(np.min(x)-1, np.max(x)+10, 100)
    s.shape = (100,1)
    
    # make line and plot it
    t = w[0] + s*w[1]
    ax.plot(s,t,'-r',linewidth = 3) 

# create cost surface
def make_cost_surface(x,y,ax):
    # make grid over which surface will be plotted
    r = np.linspace(-1,1,100)    
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    
    # generate surface based on given data - done very lazily - recomputed each time
    g = 0
    P = len(y)
    for p in range(0,P):
        g+= (s + t*x[p] - y[p])**2
        
    # reshape and plot the surface, as well as where the zero-plane is
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    g.shape = (np.size(r),np.size(r))
    ax.plot_surface(s,t,g,alpha = 0.15)
    ax.plot_surface(s,t,g*0,alpha = 0.1)
    
    # make plot look nice
    ax.view_init(40,20)
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([0,300])
    # ax.set_ylabel('b        ',fontsize = 17)
    # ax.set_xlabel('   w',fontsize = 17)
    ax.set_xlabel('intercept        ',fontsize = 17)
    ax.set_ylabel('slope        ',fontsize = 17)


    
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax.set_zlabel('g(b,w)        ',fontsize = 17, rotation = 0)
    ax.set_zlabel('cost value           ',fontsize = 17, rotation = 0)
    
# plot gradient descent step values on cost function
def plot_steps(w_history,steps_shown,cost_values,ax):
    for j in range(0,len(cost_values)):
        w_temp = w_history[:,steps_shown[j]]
        # show current step in the w-plane
        #ax.scatter(w_temp[0],w_temp[1],color = 'm',marker = 'o',linewidth = 3)
        
        # show current step evaluated at cost surface
        ax.scatter(w_temp[0],w_temp[1],cost_values[j],color = 'k',marker = 'x',linewidth = 3)

# main animation function
def animate(x,y,w_history):
    # create figure and panels
    fig = plt.figure(figsize = (16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # formulate full input data matrix X - i.e., just pad with single row of ones
    temp = np.ones((len(x),1))
    X = np.concatenate((temp,x),1)
    X = X.T
    
    # loop over gradient descent steps and plot
    cost_values = []
    steps_shown = []
    
    # print over these steps only
    a = np.arange(0,10)
    b = np.arange(10,100,10)
    steps = []
    for i in range(0,len(a)):
        steps.append(a[i])
    for i in range(0,len(b)):
        steps.append(b[i])    
    
    for k in range(0,len(steps)):
        i = steps[k]
        w = w_history[:,i]
        
        # plot points and fit
        plot_pts(x,y,ax1)
        plot_fit(x,y,w,ax1)
        
        # plot surface
        make_cost_surface(x,y,ax2)
        
        # plot current weight and cost function value
        current_val = calculate_cost_value(X,y,w)
        cost_values.append(current_val)
        steps_shown.append(i)
        plot_steps(w_history,steps_shown,cost_values,ax2)
        
        # clear display for next round - prevent new figures from being plotted
        display.clear_output(wait=True)
        display.display(plt.gcf()) 
        ax1.clear()
        ax2.clear()
        #plt.pause(0.05)  # pause if necessary 

    # prevent new figures from popping up - not sure how this works but it does
    display.clear_output(wait=True)
    
    # plot very last step
    w = w_history[:,-1]
    plot_pts(x,y,ax1)
    plot_fit(x,y,w,ax1)
    make_cost_surface(x,y,ax2)
    current_val = calculate_cost_value(X,y,w)
    cost_values.append(current_val)
    steps_shown.append(i)
    plot_steps(w_history,steps_shown,cost_values,ax2)
    
    
########## gradient descent functions ##########
# form each partial of the linear regression cost function 
def compute_derivatives(x,y,b,w):
    # initialize each partial derivative as zero
    gprime_b = 0
    gprime_w = 0
    
    # loop over points and update each partial
    P = len(y)
    for p in range(0,P):
        temp = 2*(b + w*x[p] - y[p])
        gprime_b += temp
        gprime_w += temp*x[p]
                   
    return gprime_b,gprime_w

# gradient descent function
def linear_regression_gradient_descent(x,y,max_its):        
    # initialize parameters - we choose this special to illustrate whats going on
    b = -0.1    # initial intercept
    w = -1      # initial slope
    
    # choose fixed steplength value
    alpha = 0.0005
    
    # make list to record weights at each step of algorithm
    param_history = np.zeros((2,max_its+1)) 
    param_history[0,0] = b
    param_history[1,0] = w
    
    # gradient descent loop
    for k in range(1,max_its+1):   

        # compute each partial derivative - gprime_b is partial with respect to b, gprime_w the partial with respect to w
        gprime_b,gprime_w = compute_derivatives(x,y,b,w)        
        
        # take descent step in each partial derivative
        b = b - alpha*gprime_b
        w = w - alpha*gprime_w

        # save new weights
        param_history[0,k] = b
        param_history[1,k] = w
            
    # return parameters from each gradient descent step for cool animation!
    return param_history 
