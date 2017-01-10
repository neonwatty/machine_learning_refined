import numpy as np
import matplotlib.pyplot as plt
import math
from IPython import display

# load data
def load_data(dataset):
    # a toy 2d dataset so we can plot things
    data = np.matrix(np.genfromtxt(dataset, delimiter=','))
    x = np.asarray(data[:,0:2])
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    return x,y

########## cost function and misclassification counting functions ##########
# calculate cost function at an input step
def calculate_cost_value(X,y,w):
    # define limits
    P = len(y)
    
    # loop for cost function
    cost = 0
    for p in range(0,P):
        y_p = y[p]
        x_p = X[:,p]
        temp = (1 + np.exp(-y_p*np.dot(x_p.T,w)))
        
        # update cost sum 
        cost+=np.log(temp)
    return cost

# calculate number of misclassifications 
def calculate_misclass(X,y,w):
    # loop for cost function
    num_misclass = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = int(y[p])
        yhat_p = np.sign(np.dot(x_p.T,w))
        
        if y_p != yhat_p:
            num_misclass+=1
    return num_misclass

# print out cost function values and misclassifications at each step
def show_step_info(X,y,w):
    # calculate cost function value and number of misclassifications at each step
    cost_values = np.zeros((num_steps,1))     # container for cost values
    num_misclasses = np.zeros((num_steps,1)) # container for num misclassifications
    for k in range(0,num_steps):
        # grab n^th step weights
        w = w_history[:,k]

        # calculate cost function value at n^th step
        value = calculate_cost_value(X,y,w)
        cost_values[k] = value

        # calculate number of misclassifications at n^th step
        misses = calculate_misclass(X,y,w)
        num_misclasses[k] = misses
        
    # plot both cost value and number of misclassifications at each iteration
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(121)
    ax1.plot(cost_values)
    ax1.set_xlabel('step')
    ax1.set_ylabel('cost value')
    ax1.set_title('cost funcion value at each step')

    ax2 = fig.add_subplot(122)
    ax2.plot(num_misclasses)
    ax2.set_xlabel('step')
    ax2.set_ylabel('num misclassifications')
    ax2.set_title('number of misclassifications per step')

# main plotting function for cost function / misclassifications
def plot_costvals(steps,values,ax):
    # plot cost function values
    ax.plot(steps,values,color = 'k')
    ax.plot(steps[-1],values[-1],color = [0,0,0],marker = 'o')

    # dress up plot 
    ax.set_xlabel('step',fontsize=15,labelpad = 5)
    ax.set_ylabel('cost function value',fontsize=15,rotation = 90,labelpad = 15)
    ax.set_title('cost function value at steps of gradient descent',fontsize=15)

########## plotting functions ##########
# plot 2d points
def plot_pts(x,y,ax,dim):
    # plot points
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    class_nums = np.unique(y)
    for i in range(0,len(class_nums)):
        l = np.argwhere(y == class_nums[i])
        l = l[:,0]  
        if dim == 2:
            ax.scatter(x[l,0],x[l,1], s = 50,color = color_opts[i,:],edgecolor = 'k')
        elif dim == 3:
            ax.set_zlim(-1.1,1.1)
            ax.set_zticks([-1,0,1])
            for j in range(0,len(l)):
                h = l[j]
                ax.scatter(x[h,0],x[h,1],y[h][0],s = 40,c = color_opts[i,:])
            
    # dress panel correctly
    ax.set_xlabel('$x_1$',fontsize=20,labelpad = 5)
    ax.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 15)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set(aspect = 'equal')
        
# plot simple toy data, linear separators, and fused rule
def plot_linear_rule(x,y,w,ax1):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)    
        
    # plot original dataset with separator(s)
    r = np.linspace(-0.1,1.1,500)    
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    f = w[0] + s*w[1] + t*w[2]
    z = np.sign(f)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    # plot boundary etc.,
    ax1.contour(s,t,z,colors = 'k',linewidths = 2.5)
    ax1.contourf(s,t,z+1,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))

# animate using cost function graph intead of surface itself
def animate_linear_classification(x,y,w_history):
    # formulate full input data matrix X - i.e., just pad with single row of ones
    temp = np.shape(x)  
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    
    # make figure
    fig = plt.figure(figsize = (16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    big_val = calculate_cost_value(X,y,w_history[:,0])

    # begin loop
    cost_values = []
    steps_shown = []
    for i in range(0,20,1):
        # plot separator and rules
        w = w_history[:,i]
        plot_pts(x,y,ax1,dim = 2)
        plot_linear_rule(x,y,w,ax1)

        # plot cost function value
        current_val = calculate_cost_value(X,y,w)
        cost_values.append(current_val)
        steps_shown.append(i)
        plot_costvals(steps_shown,cost_values,ax2)
        ax2.set_xlim(-1,20)
        ax2.set_ylim(0,big_val + 2)

        # clear separator / rule from middle panel
        display.clear_output(wait=True)
        display.display(plt.gcf()) 
        ax1.clear()
    display.clear_output(wait=True)
    plot_pts(x,y,ax1,dim = 2)
    plot_linear_rule(x,y,w,ax1)
    
########## gradient descent functions ##########
# compute cth class gradient for single data point
def compute_grad(X,y,w):
    # produce gradient for each class weights
    grad = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = y[p]
        grad+= -1/(1 + np.exp(y_p*np.dot(x_p.T,w)))*y_p*x_p
    
    grad.shape = (len(grad),1)
    return grad

# learn all C separators together running stochastic gradient descent
def logistic_regression(x,y,max_its):    
    # formulate full input data matrix X - i.e., just pad with single row of ones
    temp = np.shape(x)  
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    
    # initialize weights - we choose w = random for illustrative purposes
    w = np.random.randn(3,1)
    
    # choose fixed steplength value
    alpha = 0.05
    
    # make list to record weights at each step of algorithm
    w_history = np.zeros((3,max_its+1)) 
    w_history[:,0] = w.flatten()
    # gradient descent loop
    for k in range(1,max_its+1):   

        # form gradient
        grad = compute_grad(X,y,w)
        
        # take gradient descent step
        w = w - alpha*grad

        # save new weights
        w_history[:,k] = w.flatten()
    
    # return weights from each step
    return w_history    