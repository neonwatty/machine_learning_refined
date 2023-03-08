import math, time, copy

# plotting functions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D 
from IPython.display import clear_output

# import autograd functionality
from autograd import value_and_grad
import autograd.numpy as np
from autograd.misc.flatten import flatten_func

# compare cost to counting
def compare_histories(histories,**kwargs):
    # parse input args
    label1 = ''; label2 = '';
    if 'label1' in kwargs:
        label1 = kwargs['label1']
    if 'label2' in kwargs:
        label2 = kwargs['label2']  
    plot_range = len(histories[0])
    if 'plot_range' in kwargs:
        plot_range = kwargs['plot_range']

    ##### setup figure to plot #####
    # initialize figure
    fig = plt.figure(figsize = (15,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 1) 
    ax1 = plt.subplot(gs[0]); 

    # run through weights, evaluate classification and counting costs, record
    c = 1
    for history in histories:
        # plot both classification and counting cost histories
        if c == 1:
            ax1.plot(np.arange(1,len(history) + 1),history,label = label1,linewidth = 4*(0.8)**(c))
        else:
            ax1.plot(np.arange(1,len(history) + 1),history,label = label2,linewidth = 4*(0.8)**(c))
        c += 1

    ax1.set_xlabel('value of $K$',fontsize = 10)
    ax1.set_ylabel('cost function value',fontsize = 10)
    plt.legend(loc='upper right')
    ax1.set_xticks(plot_range)
    plt.show()
        
        
# compare cost histories from multiple runs
def plot_cost_histories(histories,start,**kwargs):
    # plotting colors
    colors = ['k','magenta','aqua','blueviolet','chocolate']

    # initialize figure
    fig = plt.figure(figsize = (15,4))

    # create subplot with 1 panel
    gs = gridspec.GridSpec(1, 1) 
    ax = plt.subplot(gs[0]); 

    # any labels to add?        
    labels = [' ',' ']
    if 'labels' in kwargs:
        labels = kwargs['labels']

    # plot points on cost function plot too?
    points = False
    if 'points' in kwargs:
        points = kwargs['points']

    # run through input histories, plotting each beginning at 'start' iteration
    for c in range(len(histories)):
        history = histories[c]
        label = 0
        if c == 0:
            label = labels[0]
        else:
            label = labels[1]

        # check if a label exists, if so add it to the plot
        if np.size(label) == 0:
            ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c]) 
        else:               
            ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 

        # check if points should be plotted for visualization purposes
        if points == True:
            ax.scatter(np.arange(start,len(history),1),history[start:],s = 90,color = colors[c],edgecolor = 'w',linewidth = 2,zorder = 3) 


    # clean up panel
    xlabel = 'step $k$'
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    ylabel = r'$g\left(\mathbf{w}^k\right)$'
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    ax.set_xlabel(xlabel,fontsize = 14)
    ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
    if np.size(label) > 0:
        anchor = (1,1)
        if 'anchor' in kwargs:
            anchor = kwargs['anchor']
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax.set_xlim([start - 0.5,len(history) - 0.5])

   # fig.tight_layout()
    plt.show()
    
    
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


    