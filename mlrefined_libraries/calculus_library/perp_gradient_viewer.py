# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad    
import autograd.numpy as np
from autograd import hessian 
import math
import time
from matplotlib import gridspec
import copy

'''
This method visualizes the contours of a function taking in two inputs.  Then for a set of input points
the gradient is computed (at each point) and drawn as an arrow on top of the contour plot
'''

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
    
def illustrate_gradients(g,pts,**kwargs):   
    # user defined args
    pts_max = np.max(np.max(pts)) + 3
    viewmax = max(3,pts_max)
    colors = ['lime','magenta','orangered']

    if 'viewmax' in kwargs:
        viewmax = kwargs['viewmax']

    num_contours = 15
    if 'num_contours' in kwargs:
        num_contours = kwargs['num_contours']  
        
    ##### setup figure to plot #####
    # initialize figure
    fig = plt.figure(figsize = (8,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
    ax1 = plt.subplot(gs[0]); ax1.axis('off')
    ax2 = plt.subplot(gs[1]); ax2.set_aspect('equal')
    ax3 = plt.subplot(gs[2]); ax3.axis('off')

    ### compute gradient of input function ###
    nabla_g = grad(g)
    
    # loop over points and determine levels
    num_pts = pts.shape[1]
    levels = []
    for t in range(num_pts):
        pt = pts[:,t]
        g_val = g(pt)
        levels.append(g_val)
    levels = np.array(levels)
    inds =  np.argsort(levels, axis=None)
    pts = pts[:,inds]
    levels = levels[inds]    
    
    # evaluate all input points through gradient function
    grad_pts = []
    num_pts = pts.shape[1]
    for t in range(num_pts):
        # point
        color = colors[t]
        pt = pts[:,t]
        nabla_pt = nabla_g(pt)
        nabla_pt /= np.linalg.norm(nabla_pt)
        
        # plot original points
        ax2.scatter(pt[0],pt[1],s = 80,c = color,edgecolor = 'k',linewidth = 2,zorder = 3)

        ### draw 2d arrow in right plot ###
        # create gradient vector
        grad_pt = pt - nabla_pt
        
        # plot gradient direction
        scale = 0.3
        arrow_pt = (grad_pt - pt)*0.78*viewmax*scale
        ax2.arrow(pt[0],pt[1],arrow_pt[0],arrow_pt[1], head_width=0.1, head_length=0.1, fc='k', ec='k',linewidth=4,zorder = 2,length_includes_head=True)
        ax2.arrow(pt[0],pt[1],arrow_pt[0],arrow_pt[1], head_width=0.1, head_length=0.1, fc=color, ec=color,linewidth=2.75,zorder = 2,length_includes_head=True)
        
        ### compute orthogonal line to contour ###
        # compute slope of gradient direction
        slope = float(arrow_pt[1])/float(arrow_pt[0])
        perp_slope = -1/slope
        perp_inter = pt[1] - perp_slope*pt[0]        
        
        # find points on orthog line approx 'scale' away in both directions (lazy quadratic formula)
        scale = 1.5
        s = np.linspace(pt[0] - 5, pt[0] + 5,1000)
        y2 = perp_slope*s + perp_inter
        dists = np.abs(((s - pt[0])**2 + (y2 - pt[1])**2)**0.5 - scale)
        ind = np.argmin(dists)
        x2 = s[ind]
    
        # plot tangent line to contour
        if x2 < pt[0]:
            s = np.linspace(x2,pt[0] + abs(x2 - pt[0]),200)
        else:
            s = np.linspace(pt[0] - abs(x2 - pt[0]),x2,200)

        v = perp_slope*s + perp_inter
        ax2.plot(s,v,zorder = 2,c = 'k',linewidth = 3)
        ax2.plot(s,v,zorder = 2,c = colors[t],linewidth = 1)
        
    # generate viewing range 
    contour_plot(ax2,g,pts,viewmax,num_contours,colors,levels)
    plt.show()

### visualize contour plot of cost function ###
def contour_plot(ax,g,pts,wmax,num_contours,my_colors,pts_levels):
    #### define input space for function and evaluate ####
    w1 = np.linspace(-wmax,wmax,100)
    w2 = np.linspace(-wmax,wmax,100)
    w1_vals, w2_vals = np.meshgrid(w1,w2)
    w1_vals.shape = (len(w1)**2,1)
    w2_vals.shape = (len(w2)**2,1)
    h = np.concatenate((w1_vals,w2_vals),axis=1)
    func_vals = np.asarray([g(s) for s in h])
    w1_vals.shape = (len(w1),len(w1))
    w2_vals.shape = (len(w2),len(w2))
    func_vals.shape = (len(w1),len(w2)) 

    ### make contour right plot - as well as horizontal and vertical axes ###
    # set level ridges
    levelmin = min(func_vals.flatten())
    levelmax = max(func_vals.flatten())
    cutoff = 0.3
    cutoff = (levelmax - levelmin)*cutoff
    numper = 3
    levels1 = np.linspace(cutoff,levelmax,numper)
    num_contours -= numper

    ##### plot filled contours with generic contour lines #####
    # produce generic contours
    levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
    levels = np.unique(np.append(levels1,levels2))
    num_contours -= numper
    while num_contours > 0:
        cutoff = levels[1]
        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels2,levels))
        num_contours -= numper
    
    # plot the contours
    ax.contour(w1_vals, w2_vals, func_vals,levels = levels[1:],colors = 'k')
    ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
    
    ###### add contour curves based on input points #####
    # add to this list the contours passing through input points
    ax.contour(w1_vals, w2_vals, func_vals,levels = pts_levels,colors = 'k',linewidths = 3)
    ax.contour(w1_vals, w2_vals, func_vals,levels = pts_levels,colors = my_colors, linewidths = 2.5)

    ###### clean up plot ######
    ax.set_xlabel('$w_0$',fontsize = 12)
    ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
    ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
    ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)