import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from sklearn import preprocessing
from IPython import display
 
# load data
def load_data(dataset):
    data = np.matrix(np.genfromtxt(dataset, delimiter=','))
    x = np.asarray(data[:,0:-1])
    y = np.asarray(data[:,-1])
    return x,y

########## feature transformation functions ##########
def poly_features(X,D):
    # just use scikit-learn's preprocessing for this - faster
    polyfit = preprocessing.PolynomialFeatures(D)
    F = polyfit.fit_transform(X)
   
    return F
    
def fourier_features(X,D):
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = (D+1)**2 + 1
    F = np.zeros((P,M))
    
    # loop over dataset, transforming each input data to fourier feature
    for p in range(0,P):
        f = np.zeros((1,M))
        x = X[p,:]
        m = 0
        
        # enumerate all individual Fourier terms - probably enumerating in complex exponential form is best. 
        for i in range(0,D+1):
            for j in range(0,D+1):
                F[p][m] = np.cos(i*x[0])*np.sin(j*x[1])
                m+=1
    return F
        
# generate and save parameters for random features
def make_random_params(X,D):
    # sizes of things
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = D
    
    # make random projections and save them
    R = np.random.randn(M,N+1)
    np.savetxt('random_projections.txt', R)    

# generate random features
def random_features(X,D):
    # create random projections for feature transformations
    R = np.loadtxt('random_projections.txt')
    
    # sizes of things
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = D
    F = np.zeros((P,M+1))  

    # set external biases, then tranform random projection of each point
    for p in range(0,P):   
        F[p,0] = 1
        x_p = X[p,:]
        for m in range(0,M):
            r = R[m,:]
            
            # compute random projection of x_p onto r
            proj = r[0] + np.sum(x_p*r[1:])
            
            # take nonlinear transformation of random projection
            # using cosine right now
            F[p,m+1] = np.cos(proj)
            
    return F

# this is the main switch that routes choice of feeatype to its proper function above
def create_features(X,D,feat_type):
    F = 0
    # make desired feature type
    if feat_type == 'poly':
        F = poly_features(X,D)
    if feat_type == 'fourier':
        F = fourier_features(X,D)
    if feat_type == 'random': 
        F = random_features(X,D)
     
    return F

########## optimization functions ##########
# make your own exponential function that ignores cases where exp = inf
def my_exp(val):
    newval = 0
    if val > 100:
        newval = np.inf
    if val < -100:
        newval = 0
    if val < 100 and val > -100:
        newval = np.exp(val)
    return newval

# calculate grad and Hessian for newton's method
def compute_grad_and_hess(X,y,w):
    hess = 0
    grad = 0
    for p in range(0,len(y)):
        # precompute
        x_p = X[:,p]
        y_p = y[p]
        s = 1/(1 + my_exp(y_p*np.dot(x_p.T,w)))
        g = s*(1-s)
        
        # update grad and hessian
        grad+= -s*y_p*x_p
        hess+= np.outer(x_p,x_p)*g
        
    grad.shape = (len(grad),1)
    return grad,hess

# note here you are loading *transformed features* as X
def softmax_2class_newton(X,y,max_its):

    # initialize variables
    N,P = np.shape(X)
    w = np.random.randn(N,1)

    # record number of misclassifications on training set at each epoch 
    w_history = np.zeros((N,max_its+1)) 
    w_history[:,0] = w.flatten()
    # outer descent loop
    for k in range(1,max_its+1):

        # compute gradient and Hessian
        grad,hess = compute_grad_and_hess(X,y,w)
        
        # take Newton method step
        temp = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),temp)
        
        # update misclass container and associated best W
        w_history[:,k] = w.flatten()
        
    # return goodies
    return w_history  

########## calculate cost function values and number of misclassifications ##########
# calculate cost function at an input step
def calculate_cost_value(X,y,w):
    # define limits
    P = len(y)
    
    # loop for cost function and add em up
    cost = 0
    for p in range(0,P):
        y_p = y[p]
        x_p = X[:,p]
        temp = (1 + np.exp(-y_p*np.dot(x_p.T,w)))
        
        # update cost sum 
        cost+=np.log(temp)
    return cost

# calculate number of misclassifications value for a given input weight W=[w_1,...,w_C]
def calculate_misclass(X,y,w):
    # loop for cost function and add em up
    num_misclass = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = int(y[p])
        yhat_p = np.sign(np.dot(x_p.T,w))
        
        if y_p != yhat_p:
            num_misclass+=1
    return num_misclass
                         
# simple plotting function for cost function / num misclassifications
def plot_costvals(steps,values,ax):
    # plot cost function values
    ax.plot(steps,values,color = 'k')
    ax.plot(steps[-1],values[-1],color = [0,0,0],marker = 'o')

    # dress up plot 
    ax.set_xlabel('step',fontsize=15,labelpad = 5)
    ax.set_ylabel('cost function value',fontsize=15,rotation = 90,labelpad = 15)
    ax.set_title('cost function value at steps of Newtons method',fontsize=15)
                
# print out both cost function values and misclassifications at each step
def show_step_info(X,y,w):
    # calculate cost function value and number of misclassifications at each step
    cost_values = np.zeros((num_steps,1))     # container for cost values
    num_misclasses = np.zeros((num_steps,1)) # container for num misclassifications
    for k in range(0,num_steps):
        # grab k^th step weights
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

########## plotting function ##########
# plot 2d points
def plot_pts(x,y,ax,dim):
    # some nice specialty colors!
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    
    # a simple switch between 2d and 3d data
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
            
    # dress panel correctly with axis labels etc.
    ax.set_xlabel('$x_1$',fontsize=20,labelpad = 5)
    ax.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 15)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
# plot simple toy data, nonlinear separators, and fused rule        
def plot_nonlinear_rule(x,y,w,feat_type,D,ax):
    ## initialize figure, plot data, and dress up panels with axes labels etc.,
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)    
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])

    ## plot separator(s), and fuse individual subproblem separators into one joint rule
    r = np.linspace(-0.1,1.1,200)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    
    # transform data points into basis features
    h = np.concatenate((s,t),1)
    h = create_features(h,D,feat_type)
    f = np.dot(h,w)
    z = (np.sign(f) + 1)

    # show rule 
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))
    ax.contour(s,t,z,colors = 'k',linewidths = 2.5)
    ax.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))

# main plotting function 
def animate_nonlinear_classification(x,y,w_history,X,feat_type,degree,max_its):
    
    # make figure and panels
    fig = plt.figure(figsize = (16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    big_val = calculate_cost_value(X,y,w_history[:,0])
    
    # begin loop
    cost_values = []
    steps_shown = []
    for i in range(0,max_its+1):
        # plot separator and rules
        w = w_history[:,i]
        plot_pts(x,y,ax1,dim = 2)
        plot_nonlinear_rule(x,y,w,feat_type,degree,ax1)

        # plot cost function value
        current_val = calculate_cost_value(X,y,w)
        if current_val > 100000:
            current_val = 100000
        cost_values.append(current_val)
        steps_shown.append(i+1)
        plot_costvals(steps_shown,cost_values,ax2)
        ax2.set_xlim(-1,max_its + 1)
        ax2.set_ylim(-1,big_val + 2)

        # clear separator / rule from middle panel
        display.clear_output(wait=True)
        display.display(plt.gcf())                 
        ax1.clear()
        
                
    # this next line both clears the panels and also prevents a bunch from printing out (a new figure for each new plot in the loop) - I have no idea why it does this, but it does, and it works
    
    display.clear_output(wait=True)
    plot_pts(x,y,ax1,dim = 2)
    plot_nonlinear_rule(x,y,w,feat_type,degree,ax1)
    

########## main function ##########
def run(dataset,feat_type,degree):
    # load data
    x,y = load_data(dataset)
  
    # make random weights and save if feattype == 'random' is chosen
    if feat_type == 'random':
        make_random_params(x,degree)
        
    # formulate full input data matrix X - i.e., just pad with single row of ones
    X = create_features(x,degree,feat_type)
    X = X.T
    
    # run Newton's method
    max_its = 10
    w_history = softmax_2class_newton(X,y,max_its)

    # animate each step (or each 2nd, 3rd, etc., step)
    animate_nonlinear_classification(x,y,w_history,X,feat_type,degree,max_its)