import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from autograd import numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history,cost_history
    
# plot multi-output regression dataset where output dimension C = 2
def plot_data(x,y,view1,view2):    
    # construct panels
    fig = plt.figure(figsize = (9,4))
    ax0 = plt.subplot(121,projection='3d')
    ax0.view_init(view1[0],view1[1])
    ax0.axis('off')

    ax1 = plt.subplot(122,projection='3d')
    ax1.view_init(view2[0],view2[1])
    ax1.axis('off')

    # scatter plot data in each panel
    ax0.scatter(x[0,:],x[1,:],y[0,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    ax1.scatter(x[0,:],x[1,:],y[1,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    plt.show()
   
# plot multi-output regression dataset with fits provided by 'predictor'
def plot_regressions(x,y,predictor,view1,view2):        
    # import all the requisite libs
    # construct panels
    fig = plt.figure(figsize = (9,4))
    ax0 = plt.subplot(121,projection='3d')
    ax0.view_init(view1[0],view1[1])
    ax0.axis('off')

    ax1 = plt.subplot(122,projection='3d')
    ax1.view_init(view2[0],view2[1])
    ax1.axis('off')

    # scatter plot data in each panel
    ax0.scatter(x[0,:],x[1,:],y[0,:],c='k',edgecolor = 'w',linewidth = 1,s=60)
    ax1.scatter(x[0,:],x[1,:],y[1,:],c='k',edgecolor = 'w',linewidth = 1,s=60)

    # construct input for each model fit
    a_ = np.linspace(0,1,15)
    a,b = np.meshgrid(a_,a_)
    a = a.flatten()[np.newaxis,:]
    b = b.flatten()[np.newaxis,:]
    c = np.vstack((a,b))

    # evaluate model 
    p = predictor(c)
    m1 = p[0,:]
    m2 = p[1,:]

    # plot each as surface
    a.shape = (a_.size,a_.size)
    b.shape = (a_.size,a_.size)
    m1.shape = (a_.size,a_.size)
    m2.shape = (a_.size,a_.size)

    ax0.plot_surface(a,b,m1,alpha = 0.25,color = 'lime',cstride = 2,rstride = 2,linewidth = 1,edgecolor ='k')
    ax1.plot_surface(a,b,m2,alpha = 0.25,color = 'lime',cstride = 2,rstride = 2,linewidth = 1,edgecolor ='k')

    plt.show()