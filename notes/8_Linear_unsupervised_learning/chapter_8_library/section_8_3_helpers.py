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


def pca_visualizer(X,W,pcs):
    # renderer    
    fig = plt.figure(figsize = (15,5))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2) 
    ax1 = plt.subplot(gs[0],aspect = 'equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
                 
    # sphere the results
    ars = np.eye(2)
        
    # loop over panels and plot each 
    c = 1
    for ax,pt,ar in zip([ax1,ax2],[X,W],[pcs,ars]): 
        # set viewing limits for originals
        xmin = np.min(pt[0,:])
        xmax = np.max(pt[0,:])
        xgap = (xmax - xmin)*0.15
        xmin -= xgap
        xmax += xgap
        ymin = np.min(pt[1,:])
        ymax = np.max(pt[1,:])
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
    
        # scatter points
        ax.scatter(pt[0,:],pt[1,:],s = 60, c = 'k',edgecolor = 'w',linewidth = 1,zorder = 2)
   
        # plot original vectors
        vector_draw(ar[:,0].flatten(),ax,color = 'red',zorder = 3)
        vector_draw(ar[:,1].flatten(),ax,color = 'red',zorder = 3)

        # plot x and y axes, and clean up
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k', linewidth=1.5,zorder = 1)
        ax.axvline(x=0, color='k', linewidth=1,zorder = 1)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.grid('off')

        # set tick label fonts
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        
        # plot title
        if c == 1:
            ax.set_title('original space',fontsize = 22)
            ax.set_xlabel(r'$x_1$',fontsize = 22)
            ax.set_ylabel(r'$x_2$',fontsize = 22,rotation = 0,labelpad = 10)
        if c == 2:
            ax.set_title('PCA transformed space',fontsize = 22)
            ax.set_xlabel(r'$v_1$',fontsize = 22)
            ax.set_ylabel(r'$v_2$',fontsize = 22,rotation = 0,labelpad = 10)
        c+=1
        
# set axis in left panel
def move_axis_left(ax):
    tmp_planes = ax.zaxis._PLANES 
    ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                        tmp_planes[0], tmp_planes[1], 
                        tmp_planes[4], tmp_planes[5])   
    ax.grid(False)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    
def plot_hyperplane(data,slopes,ax):
    # define input space
    xmin = np.min(data[:,0])
    xmax = np.max(data[:,0])
    xgap = (xmax - xmin)*0.1
    xmin -= xgap
    xmax += xgap
    
    ymin = np.min(data[:,1])
    ymax = np.max(data[:,1])
    ygap = (ymax - ymin)*0.1
    ymin -= ygap
    ymax += ygap
    
    # create meshgrid
    xrange = np.linspace(xmin,xmax,200)
    yrange = np.linspace(ymin,ymax,200)
    w1_vals, w2_vals = np.meshgrid(xrange,yrange)
    w1_vals.shape = (len(xrange)**2,1)
    w2_vals.shape = (len(yrange)**2,1)
    
    # compute normal vector to plane
    normal_vector = np.cross(slopes[:,0], slopes[:,1])
    normal_vector = normal_vector/(-normal_vector[-1])
    
    # hyperplane function
    func = lambda w: normal_vector[0]*w[0] + normal_vector[1]*w[1]

    # evaluate hyperplane
    zvals = func([w1_vals,w2_vals]) 

    # vals for cost surface, reshape for plot_surface function
    w1_vals.shape = (len(xrange),len(xrange))
    w2_vals.shape = (len(yrange),len(yrange))
    zvals.shape = (len(xrange),len(yrange))

    ### plot function and z=0 for visualization ###
    ax.plot_surface(w1_vals, w2_vals, zvals, alpha = 0.1,color = 'r',zorder = 2)
    
    
def project_data_from_3d_to_2d(X,C,view):

    # create plotting panel
    fig = plt.figure(figsize = (15,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5,1,1.5]) 
    #ax1 = plt.subplot(gs[0],projection='3d',aspect = 'equal');  
    ax1 = plt.subplot(gs[0],projection='3d'); 
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    #ax3 = plt.subplot(gs[2],projection='3d',aspect = 'equal');  
    ax3 = plt.subplot(gs[2],projection='3d'); 

    #### plot original data ####
    # scatter normalized data
    ax1.scatter(X[0,:],X[1,:],X[2,:],c = 'k',alpha = 0.25)

    # plot principal components
    a = np.zeros((2,1))
    ax1.quiver(a,a,a,C[0,:],C[1,:],C[2,:],color = 'r')

    # draw hyperplane
    plot_hyperplane(X.T,C,ax1)
    
    # clean up panel 1
    ax1.view_init(view[0],view[1])
    ax1.set_xlabel(r'$x_1$',fontsize = 18,labelpad = 5)
    ax1.set_ylabel(r'$x_2$',fontsize = 18,labelpad = 5)
    ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax1.set_zlabel(r'$x_3$',fontsize = 18,rotation = 0)
    ax1.set_title('Original data',fontsize = 20)
    move_axis_left(ax1)

    #### plot encoded data ####
    W = np.linalg.solve(np.dot(C.T,C),np.dot(C.T,X))

    # in new coordinate system defined by pcs
    ax2.scatter(W[0,:],W[1,:],c = 'k',edgecolor = 'w',linewidth = 1,s = 50,zorder = 2)

    # paint arrows on data
    ax2.arrow(0,0,0,1,fc="r", ec="r",head_width=0.15, head_length=0.15,linewidth = 2,zorder = 3)
    ax2.arrow(0,0,1,0,fc="r", ec="r",head_width=0.15, head_length=0.15,linewidth = 2,zorder = 3)   
    
    # clean up panel 2
    ax2.set_xlabel(r'$c_1$',fontsize = 18)
    ax2.set_ylabel(r'$c_2$',fontsize = 18,rotation = 0)
    ax2.axhline(y=0, color='k', linewidth=1.5,zorder = 1)
    ax2.axvline(x=0, color='k', linewidth=1,zorder = 1)
    ax2.set_title('Encoded data',fontsize = 20)
    
    xmin = np.min([-1.5,np.min(W[0,:])])
    xmax = np.max([1.5,np.max(W[0,:])])
    xgap = (xmax - xmin)*0.2
    xmin -= xgap
    xmax += xgap
    
    ymin = np.min([-1.5,np.min(W[1,:])])
    ymax = np.max([1.5,np.max(W[1,:])])
    ygap = (ymax - ymin)*0.2
    ymin -= ygap
    ymax += ygap
   
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])
    
    #### plot decoded data ####
    # scatter decoded data
    X_d = np.dot(C,W)
    ax3.scatter(X_d[0,:],X_d[1,:],X_d[2,:],c = 'k',edgecolor = 'r',linewidth = 1,alpha = 0.25)
    
    # draw hyperplane
    plot_hyperplane(X.T,C,ax3)
    
    # clean up panel 1
    ax3.view_init(view[0],view[1])
    ax3.set_xlabel(r'$x_1$',fontsize = 18,labelpad = 5)
    ax3.set_ylabel(r'$x_2$',fontsize = 18,labelpad = 5)
    ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax3.set_zlabel(r'$x_3$',fontsize = 18,rotation = 0)
    ax3.set_title('Decoded data',fontsize = 20)
    move_axis_left(ax3)
    
    # set viewing range based on original plot
    vals = ax1.get_zlim()
    ax3.set_zlim([vals[0],vals[1]])
    
    
def visual_comparison(x,weights):
    '''
    Visually compare the results of several runs of PCA applied to two dimensional input and 
    two principal components
    '''
    # do weights
    weights = np.array(weights)
    num_runs = np.ndim(weights)
    
    # plot data
    fig = plt.figure(figsize = (15,4))
    gs = gridspec.GridSpec(1, num_runs) 
    for run in range(num_runs):
        # create subplot
        ax = plt.subplot(gs[run],aspect = 'equal'); 
        w_best = weights[run]

        # scatter data
        ax.scatter(x[0,:],x[1,:],c = 'k')
        
        # plot pc 1
        vector_draw(w_best[:,0],ax,color = 'red',zorder = 1)
        vector_draw(w_best[:,1],ax,color = 'red',zorder = 1)

        # plot vertical / horizontal axes
        ax.axhline(linewidth=0.5, color='k',zorder = 0)
        ax.axvline(linewidth=0.5, color='k',zorder = 0)
        ax.set_title('run ' + str(run + 1),fontsize=16)
        ax.set_xlabel(r'$x_1$',fontsize = 16)
        ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)

# simple plot of 2d vector addition / paralellagram law
def vector_add_plot(vec1,vec2,ax):   
    # plot each vector
    head_length = 0.5
    head_width = 0.5
    veclen = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec1_orig = copy.deepcopy(vec1)
    vec1 = (veclen - head_length)/veclen*vec1
    veclen = math.sqrt(vec2[0]**2 + vec2[1]**2)
    vec2_orig = copy.deepcopy(vec2)
    vec2 = (veclen - head_length)/veclen*vec2
    ax.arrow(0, 0, vec1[0],vec1[1], head_width=head_width, head_length=head_length, fc='b', ec='b',linewidth=2,zorder = 2)
    ax.arrow(0, 0, vec2[0],vec2[1], head_width=head_width, head_length=head_length, fc='b', ec='b',linewidth=2,zorder = 2)
     
    # plot the sum of the two vectors
    vec3 = vec1_orig + vec2_orig
    vec3_orig = copy.deepcopy(vec3)
    veclen = math.sqrt(vec3[0]**2 + vec3[1]**2)
    vec3 = (veclen - math.sqrt(head_length))/veclen*vec3
    ax.arrow(0, 0, vec3[0],vec3[1], head_width=head_width, head_length=head_length, fc='r', ec='r',linewidth=3,zorder=2)
    
    # connect them
    ax.plot([vec1_orig[0],vec3_orig[0]],[vec1_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 1)
    ax.plot([vec2_orig[0],vec3_orig[0]],[vec2_orig[1],vec3_orig[1]],linestyle= '--',c='b',zorder=2,linewidth = 1)

# draw a vector
def vector_draw(vec,ax,**kwargs):
    color = 'k'
    if 'color' in kwargs:
        color = kwargs['color']
    zorder = 3 
    if 'zorder' in kwargs:
        zorder = kwargs['zorder']
        
    veclen = math.sqrt(vec[0]**2 + vec[1]**2)
    head_length = 0.25
    head_width = 0.25
    vec_orig = copy.deepcopy(vec)
    vec = (veclen - head_length)/veclen*vec
    ax.arrow(0, 0, vec[0],vec[1], head_width=head_width, head_length=head_length, fc=color, ec=color,linewidth=3,zorder = zorder)
      
        
        
def perfect_visualize_transform_static(C,**kwargs):
    vec1 = C[:,0]
    vec2 = C[:,1]
    
    # size up vecs
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    vec1copy = copy.deepcopy(vec1)
    vec1copy.shape = (len(vec1copy),1)
    vec2copy = copy.deepcopy(vec2)
    vec2copy.shape = (len(vec2copy),1)
     
    # renderer    
    fig = plt.figure(figsize = (15,4))
    artist = fig
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2) 
    ax1 = plt.subplot(gs[0],aspect = 'equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
     
    ### create grid of points ###
    s = np.linspace(-5,5,10)
    xx,yy = np.meshgrid(s,s)
    xx.shape = (xx.size,1)
    yy.shape = (yy.size,1)
    X = np.concatenate((xx,yy),axis=1)
    X = np.flipud(X)
    
    if 'X' in kwargs:
        X = kwargs['X'].T
             
    # swing through points and compute coeffecients
    alphas = []
    for k in range(X.shape[0]):
        vec3 = X[k,:]   
        vec3.shape = (len(vec3),1)
        A = np.concatenate((vec1copy,vec2copy),axis=1)
        b = vec3
        alpha = np.linalg.solve(A,b)
        alphas.append(alpha)
         
    # set viewing limits for originals
    xmin = np.min(X[:,0])
    xmax = np.max(X[:,0])
    xgap = (xmax - xmin)*0.15
    xmin -= xgap
    xmax += xgap
    ymin = np.min(X[:,1])
    ymax = np.max(X[:,1])
    ygap = (ymax - ymin)*0.15
    ymin -= ygap
    ymax += ygap
    
    # set viewing limits for transformed space
    alpha_xmin = np.min([a[0][0] for a in alphas])
    alpha_xmax = np.max([a[0][0] for a in alphas])
    alpha_xgap = (alpha_xmax - alpha_xmin)*0.15
    alpha_xmin -= alpha_xgap
    alpha_xmin = np.min([-0.5,alpha_xmin])
    alpha_xmax += alpha_xgap
    alpha_xmax = np.max([1.5,alpha_xmax])
    alpha_ymin = np.min([a[1][0] for a in alphas])
    alpha_ymax = np.max([a[1][0] for a in alphas])
    alpha_ygap = (alpha_ymax - alpha_ymin)*0.15
    alpha_ymin -= alpha_ygap
    alpha_ymin = np.min([-0.5,alpha_ymin])
    alpha_ymax += alpha_ygap
    alpha_ymax = np.max([1.5,alpha_ymax])

    ### take pt of grid and estimate with inputs ###        
    # scatter every point up to k
    for i in range(X.shape[0]):
        # plot original point
        pt = X[i,:]
        ax1.scatter(pt[0],pt[1],s = 60, c = 'k',edgecolor = 'w',linewidth = 1)

        # plot transformed plot
        pt = alphas[i]
        ax2.scatter(pt[0],pt[1],s = 60, c = 'k',edgecolor = 'w',linewidth = 1)
 
    # plot original vectors
    vector_draw(vec1copy.flatten(),ax1,color = 'red',zorder = 1)
    vector_draw(vec2copy.flatten(),ax1,color = 'red',zorder = 1)

    # send axis to vector adder for plotting         
    vec1 = np.array([1,0]).flatten()
    vec2 = np.array([0,1]).flatten()
    vector_draw(vec1.flatten(),ax2,color = 'red',zorder = 1)
    vector_draw(vec2.flatten(),ax2,color = 'red',zorder = 1)
            
    # plot x and y axes, and clean up
    ax1.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1.5,zorder = 1)
    ax1.axvline(x=0, color='k', linewidth=1,zorder = 1)
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax1.grid('off')
    ax1.set_xlabel(r'$x_1$',fontsize = 22)
    ax1.set_ylabel(r'$x_2$',fontsize = 22,rotation = 0,labelpad = 10)
    ax1.set_title('original data',fontsize = 24)

    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k', linewidth=1.5,zorder = 1)
    ax2.axvline(x=0, color='k', linewidth=1,zorder = 1)
    ax2.set_xlim([alpha_xmin,alpha_xmax])
    ax2.set_ylim([alpha_ymin,alpha_ymax])
    ax2.grid('off')
    ax2.set_xlabel(r'$c_1$',fontsize = 22)
    ax2.set_ylabel(r'$c_2$',fontsize = 22,rotation = 0,labelpad = 10)
    ax2.set_title('encoded data',fontsize = 24)
         
    # set tick label fonts
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(12) 
             
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(12) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(12) 
        
        
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


    