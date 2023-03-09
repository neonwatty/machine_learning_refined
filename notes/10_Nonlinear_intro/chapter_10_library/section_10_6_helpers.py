import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import autograd.numpy as np
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


# compare cost histories from multiple runs
def plot_cost_histories(histories,start,**kwargs):
    # plotting colors
    colors = ['k','magenta','aqua','blueviolet','chocolate']

    # initialize figure
    fig = plt.figure(figsize = (10,3))

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
    
    
def show_encode_decode(x,cost_history,weight_history,**kwargs):
    '''
    Examine the results of linear or nonlinear PCA / autoencoder to two-dimensional input.
    Four panels are shown: 
    - original data (top left panel)
    - data projected onto lower dimensional curve (top right panel)
    - lower dimensional curve (lower left panel)
    - vector field illustrating how points in space are projected onto lower dimensional curve (lower right panel)
    
    Inputs: 
    - x: data
    - encoder: encoding function from autoencoder
    - decoder: decoding function from autoencoder
    - cost_history/weight_history: from run of gradient descent minimizing PCA least squares
    
    Optinal inputs:
    - show_pc: show pcs?   Only useful really for linear case.
    - scale: for vector field / quiver plot, adjusts the length of arrows in vector field
    '''
    # user-adjustable args
    encoder = lambda a,b: np.dot(b.T,a)
    decoder = lambda a,b: np.dot(b,a)
    if 'encoder' in kwargs:
        encoder = kwargs['encoder']
    if 'decoder' in kwargs:
        decoder = kwargs['decoder']
    projmap = False
    if 'projmap' in kwargs:
        projmap = kwargs['projmap']
    show_pc = False
    if 'show_pc' in kwargs:
        show_pc = kwargs['show_pc']
    scale = 14
    if 'scale' in kwargs:
        scale = kwargs['scale']
    encode_label = ''
    if 'encode_label' in kwargs:
        encode_label = kwargs['encode_label']

    # pluck out best weights
    ind = np.argmin(cost_history)
    w_best = weight_history[ind]
    num_params = 0
    if type(w_best)==list:
        num_params = len(w_best)
    else:
        num_params = np.ndim(w_best) - 1

    ###### figure 1 - original data, encoded data, decoded data ######
    fig = plt.figure(figsize = (15,10))
    gs = gridspec.GridSpec(1, 3) 
    ax1 = plt.subplot(gs[0],aspect = 'equal'); 
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    ax3 = plt.subplot(gs[2],aspect = 'equal'); 

    # scatter original data with pc
    ax1.scatter(x[0,:],x[1,:],c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')

    if show_pc == True:
        for pc in range(np.shape(w_best)[1]):
            ax1.arrow(0, 0, w_best[0,pc], w_best[1,pc], head_width=0.25, head_length=0.5, fc='k', ec='k',linewidth = 4)
            ax1.arrow(0, 0, w_best[0,pc], w_best[1,pc], head_width=0.25, head_length=0.5, fc='r', ec='r',linewidth = 3)

    ### plot encoded and decoded data ###
    v = 0
    p = 0
    if num_params == 2:
        # create encoded vectors
        v = encoder(x,w_best[0])

        # decode onto basis
        p = decoder(v,w_best[1])
    else:
        # create encoded vectors
        v = encoder(x,w_best)

        # decode onto basis
        p = decoder(v,w_best)

    # plot encoded data 
    if v.shape[0] == 1:
        z = np.zeros((1,np.size(v)))
        ax2.scatter(v,z,c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')
    elif v.shape[0] == 2:
        ax2.scatter(v[0],v[1].flatten(),c = 'k',s = 60,linewidth = 0.75,edgecolor = 'w')

    
    # plot decoded data 
    ax3.scatter(p[0,:],p[1,:],c = 'k',s = 60,linewidth = 0.75,edgecolor = 'r')

    # clean up panels
    xmin1 = np.min(x[0,:])
    xmax1 = np.max(x[0,:])
    xmin2 = np.min(x[1,:])
    xmax2 = np.max(x[1,:])
    xgap1 = (xmax1 - xmin1)*0.2
    xgap2 = (xmax2 - xmin2)*0.2
    xmin1 -= xgap1
    xmax1 += xgap1
    xmin2 -= xgap2
    xmax2 += xgap2
    
    for ax in [ax1,ax2,ax3]:
        if ax == ax1 or ax == ax3:
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_xlabel(r'$x_1$',fontsize = 16)
            ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)
            ax.axvline(linewidth=0.5, color='k',zorder = 0)
        else:
            ax.set_ylim([-1,1])
            if len(encode_label) > 0:
                ax.set_xlabel(encode_label,fontsize = 16)
        ax.axhline(linewidth=0.5, color='k',zorder = 0)
    
    ax1.set_title('original data',fontsize = 18)
    ax2.set_title('encoded data',fontsize = 18)
    ax3.set_title('decoded data',fontsize = 18)
    
    # plot learned manifold
    a = np.linspace(xmin1,xmax1,400)
    b = np.linspace(xmin2,xmax2,400)
    s,t = np.meshgrid(a,b)
    s.shape = (1,len(a)**2)
    t.shape = (1,len(b)**2)
    z = np.vstack((s,t))
    
    v = 0
    p = 0
    if num_params == 2:
        # create encoded vectors
        v = encoder(z,w_best[0])

        # decode onto basis
        p = decoder(v,w_best[1])
    else:
        # create encoded vectors
        v = encoder(z,w_best)

        # decode onto basis
        p = decoder(v,w_best)
    
    ax3.scatter(p[0,:],p[1,:],c = 'k',s = 1.5,edgecolor = 'r',linewidth = 1,zorder = 0)
    ax3.axis('off')
    # set whitespace
    #fgs.update(wspace=0.01, hspace=0.5) # set the spacing between axes. 
        
    ##### bottom panels - plot subspace and quiver plot of projections ####
    if projmap == True:
        fig = plt.figure(figsize = (15,15))
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax1.scatter(p[0,:],p[1,:],c = 'r',s = 9.5)
        ax1.scatter(p[0,:],p[1,:],c = 'k',s = 1.5)
        
        ### create quiver plot of how data is projected ###
        new_scale = 0.75
        a = np.linspace(xmin1 - xgap1*new_scale,xmax1 + xgap1*new_scale,20)
        b = np.linspace(xmin2 - xgap2*new_scale,xmax2 + xgap2*new_scale,20)
        s,t = np.meshgrid(a,b)
        s.shape = (1,len(a)**2)
        t.shape = (1,len(b)**2)
        z = np.vstack((s,t))
        
        v = 0
        p = 0
        if num_params == 2:
            # create encoded vectors
            v = encoder(z,w_best[0])

            # decode onto basis
            p = decoder(v,w_best[1])
        else:
            # create encoded vectors
            v = encoder(z,w_best)

            # decode onto basis
            p = decoder(v,w_best)


        # get directions
        d = []
        for i in range(p.shape[1]):
            dr = (p[:,i] - z[:,i])[:,np.newaxis]
            d.append(dr)
        d = 2*np.array(d)
        d = d[:,:,0].T
        M = np.hypot(d[0,:], d[1,:])
        ax1.quiver(z[0,:], z[1,:], d[0,:], d[1,:],M,alpha = 0.5,width = 0.01,scale = scale,cmap='autumn') 
        ax1.quiver(z[0,:], z[1,:], d[0,:], d[1,:],edgecolor = 'k',linewidth = 0.25,facecolor = 'None',width = 0.01,scale = scale) 

        #### clean up and label panels ####
        for ax in [ax1]:
            ax.set_xlim([xmin1 - xgap1*new_scale,xmax1 + xgap1*new_scale])
            ax.set_ylim([xmin2 - xgap2*new_scale,xmax2 + xgap1*new_scale])
            ax.set_xlabel(r'$x_1$',fontsize = 16)
            ax.set_ylabel(r'$x_2$',fontsize = 16,rotation = 0,labelpad = 10)

        ax1.set_title('projection map',fontsize = 18)
        #ax.axvline(linewidth=0.5, color='k',zorder = 0)
        #ax.axhline(linewidth=0.5, color='k',zorder = 0)

        # set whitespace
        gs.update(wspace=0.01, hspace=0.5) # set the spacing between axes. 
        #ax.set_xlim([xmin1,xmax1])
        #ax.set_ylim([xmin2,xmax2])
        ax.axis('off')
    
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