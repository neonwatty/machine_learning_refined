#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

# load data
def load_data(csvfile):
    # a toy 2d dataset so we can plot things
    data = np.matrix(np.genfromtxt(csvfile, delimiter=','))
    x = np.asarray(data[:,0:2])
    y = np.asarray(data[:,2])
    y.shape = (np.size(y),1)
    
    # if 2 class change label values to proper 1...C labels 
    num_classes = len(np.unique(y))
    if num_classes == 2:
        ind = np.where(y == -1)
        y[ind] = 2

    return x,y

# calculate number of misclassifications value for a given input weight W=[w_1,...,w_C]
def calculate_misclass(X,y,W):
    # define limits
    P = len(y)
    
    # loop for cost function
    num_misclass = 0
    for p in range(0,P):
        p_c = int(y[p])-1
        guess = np.argmax(np.dot(X[:,p].T,W))
        if p_c != guess:
            num_misclass+=1
    return num_misclass

# calculate the objective value of the softmax multiclass cost for a given input weight W=[w_1,...,w_C]
def calculate_cost_value(X,y,W):
    # define limits
    P = len(y)
    C = len(np.unique(y))
    
    # loop for cost function
    cost = 0
    for p in range(0,P):
        p_c = int(y[p])-1
        temp = 0
        
        # produce innner sum
        for j in range(0,C):
            temp += np.exp(np.dot(X[:,p].T,(W[:,j] - W[:,p_c])))

        # update outer sum 
        cost+=np.log(temp)
    return cost

########## plotting functions ##########
# main plotting function for cost function / misclassifications
def plot_costvals(steps,values,ax):
    # plot cost function values
    ax.plot(steps,values,color = 'k')
    ax.plot(steps[-1],values[-1],color = [0,0,0],marker = 'o')

    # dress up plot 
    ax.set_xlabel('step',fontsize=15,labelpad = 5)
    ax.set_ylabel('cost function value',fontsize=15,rotation = 90,labelpad = 2)
    ax.set_title('cost function value at steps of gradient descent',fontsize=15)
        
# plot 2d points
def plot_pts(x,y,ax):
    # plot points
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[1, 0, 1]])
    class_nums = np.unique(y)
    for i in range(0,len(class_nums)):
        l = np.argwhere(y == class_nums[i])
        l = l[:,0]  
        ax.scatter(x[l,0],x[l,1], s = 50,color = color_opts[i,:],edgecolor = 'k')
       
    # dress panel correctly
    ax.set_xlabel('$x_1$',fontsize=20,labelpad = 5)
    ax.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 15)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set(aspect = 'equal')
    
# plot simple toy data, linear separators, and fused rule
def plot_linear_rules(x,y,W,ax):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)    
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[1, 0, 1]])
    
    # fuse individual subproblem separators into one joint rule
    r = np.linspace(-0.1,1.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((np.ones((np.size(s),1)),s,t),1)
    f = np.dot(W.T,h.T)
    z = np.argmax(f,0)+1

    # produce rule surface
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    # plot fused rule
    ax.contour(s,t,z,colors = 'k',linewidths = 2.5)
    ax.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))     
        
# animate using cost function graph intead of surface itself
def animate(x,y,W_history):
    # formulate full input data matrix X - i.e., just pad with single row of ones
    temp = np.shape(x)  
    temp = np.ones((temp[0],1))
    X = np.concatenate((temp,x),1)
    X = X.T
    
    # make figure
    fig = plt.figure(figsize = (16,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    big_val = calculate_cost_value(X,y,W_history[:,:,0])
    small_val = calculate_cost_value(X,y,W_history[:,:,np.shape(W_history)[2]-1])
    
    # begin loop
    cost_values = []
    steps_shown = []
    for i in range(0,np.shape(W_history)[2],1):
        # plot separator and rules
        W = W_history[:,:,i]       
        plot_pts(x,y,ax1)
        plot_pts(x,y,ax2)
        plot_linear_rules(x,y,W,ax2)

        # plot cost function value
        current_val = calculate_cost_value(X,y,W)
        cost_values.append(current_val)
        steps_shown.append(i)
        plot_costvals(steps_shown,cost_values,ax3)
        ax3.set_xlim(-1,np.shape(W_history)[2])
        ax3.set_ylim(small_val - 2,big_val + 2)
        
        # clear separator / rule from middle panel
        display.clear_output(wait=True)
        display.display(plt.gcf()) 
        ax1.clear()
        ax2.clear()

    display.clear_output(wait=True)
    plot_pts(x,y,ax1)
    plot_pts(x,y,ax2)
    plot_linear_rules(x,y,W,ax2)
