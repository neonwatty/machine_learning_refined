import numpy as np
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import math
import time

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########## data generating functions ##########
# make a gaussian dataset in 2-dimensions
def generate_2d_data(num_pts):
    x_1 = 3*np.random.randn(num_pts,1)
    x_2 = 1*np.random.randn(num_pts,1)

    # concatenate data into single matrix
    X = np.concatenate((x_1,x_2),axis = 1).T
    
    # rotate a bit
    theta = -np.pi*0.25
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rotation_matrix = np.reshape(rotation_matrix,(2,2))
    X = np.dot(rotation_matrix,X)
    
    # normalize data
    X_means = np.mean(X,axis=1)
    X_stds = np.std(X,axis = 1)
    X_normalized = ((X.T - X_means)/(X_stds + 10**(-7))).T
    
    return X,X_normalized

# make a gaussian dataset in 3-dimensions
def generate_3d_data(num_pts):
    x_1 = 0.75*np.random.randn(num_pts,1) + 1
    x_2 = 0.75*np.random.randn(num_pts,1) + 1
    x_3 = 7*x_1 + x_2 + 0.35*np.random.randn(num_pts,1)

    # concatenate data into single matrix
    X = np.concatenate((x_1,x_2),axis = 1)
    X = np.concatenate((X,x_3),axis=1).T 
    
    return X

########## data generating functions ##########
def frame_3d_plot(data,ax):
    # strip off each dimension of data
    x_1 = data[:,0]
    x_2 = data[:,1]
    x_3 = data[:,2]
    
    # hack to set aspect ratio to 'equal' in matplotlib 3d plot
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x_1.max()-x_1.min(), x_2.max()-x_2.min(), x_3.max()-x_3.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x_1.max()+x_1.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(x_2.max()+x_2.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(x_3.max()+x_3.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
        
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
    fig = plt.figure(figsize = (10,4))

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5,1,1.5]) 
    ax1 = plt.subplot(gs[0],projection='3d',aspect = 'equal');  
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    ax3 = plt.subplot(gs[2],projection='3d',aspect = 'equal');  

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
    
# func,
def pca_visualizer(X,W,pcs):
    # renderer    
    fig = plt.figure(figsize = (10,5))
    
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
    
# func,
def sphereing_visualizer(X,V,W,S):
    # renderer    
    fig = plt.figure(figsize = (10,5))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 3) 
    ax1 = plt.subplot(gs[0],aspect = 'equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
    ax3 = plt.subplot(gs[2],aspect = 'equal'); 
    ars2 = np.eye(2)
    ars = ars2
        
    # loop over panels and plot each 
    c = 1
    for ax,pt,ar in zip([ax1,ax2,ax3],[X,W,S],[V,ars,ars2]): 
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
        if c == 3:
            ax.set_title('Sphered data space',fontsize = 22)
            ax.set_xlabel(r'$\frac{1}{d_1^{^1/_2}}v_1$',fontsize = 22)
            ax.set_ylabel(r'$\frac{1}{d_2^{^1/_2}}v_2$',fontsize = 22,rotation = 0,labelpad = 10)
        c+=1
 
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