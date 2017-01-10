#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import my_utilities
import my_kernel_utilities

# plot simple toy classification datasets
def plot_toydata(x,y):
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,1,facecolor = 'white',figsize=(12, 5))
    plot_pts(x,y,axs,color_opts,dim = 2)
    
# plot 2d points
def plot_pts(x,y,ax,color_opts,dim):
    # plot points
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
    ax.set_xlim(np.min(x[:,0]) - 0.1,np.max(x[:,0]) + 0.1)
    ax.set_ylim(np.min(x[:,1]) - 0.1,np.max(x[:,1]) + 0.1)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set(aspect = 'equal')
    # fig.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)    
        
        
# plot simple toy data, linear separators, and fused rule
def plot_toydata_wlinear_rules(x,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)    
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    num_panels = min(num_classes,3)
    
    # make panels
    fig = plt.figure(facecolor = 'white',figsize=(20, 5))
    
    # plot original dataset
    ax1 = fig.add_subplot(1, 3, 1)
    plot_pts(x,y,ax1,color_opts,dim = 2)
  
    # plot original dataset with separator(s)
    ax2 = fig.add_subplot(1, 3, 2)
    plot_pts(x,y,ax2,color_opts,dim = 2)
    r = np.linspace(-0.1,1.1,150)

    # fuse individual subproblem separators into one joint rule
    ax3 = 0
    if num_classes == 2:
        ax3 = fig.add_subplot(1, 3, 3,projection = '3d')
        plot_pts(x,y,ax3,color_opts,dim = 3)

    else:
        ax3 = fig.add_subplot(1, 3, 3)
        plot_pts(x,y,ax3,color_opts,dim = 2)
    
    r = np.linspace(-0.1,1.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((np.ones((np.size(s),1)),s,t),1)
    f = np.dot(W.T,h.T)
    z = (np.sign(f) + 1)
    if num_classes > 2:
        z = np.argmax(f,0)+1

    # produce rule surface
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))
    if num_classes == 2:
        ax2.contour(s,t,z,colors = 'k',linewidths = 2.5)
        ax2.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))
        ax3.plot_surface(s,t,z-1,cmap = 'gray',alpha = 0.05)
        ax3.view_init(25,-60)
    else:
        ax3.contour(s,t,z,colors = 'k',linewidths = 2.5)
        ax3.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))     

# plot simple toy data, nonlinear separators, and fused rule        
def plot_toydata_wnonlinear_rules(x,y,W,feat_type,D,raw_or_kernel):
    ## initialize figure, plot data, and dress up panels with axes labels etc.,
    class_nums = np.unique(y)
    num_classes = np.size(class_nums)    
    color_opts = np.array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    num_panels = min(num_classes,3)
    
    ## make panels
    fig = plt.figure(facecolor = 'white',figsize=(20, 5))
    
    ## plot original dataset
    ax1 = fig.add_subplot(1, 3, 1)
    plot_pts(x,y,ax1,color_opts,dim = 2)
  
    ## plot original dataset in 2d and possibly 3d
    ax2 = fig.add_subplot(1, 3, 2)
    plot_pts(x,y,ax2,color_opts,dim = 2)

    ax3 = 0
    if num_classes == 2:
        ax3 = fig.add_subplot(1, 3, 3,projection = '3d')
        plot_pts(x,y,ax3,color_opts,dim = 3)

    else:
        ax3 = fig.add_subplot(1, 3, 3)
        plot_pts(x,y,ax3,color_opts,dim = 2)

    ## plot separator(s), and fuse individual subproblem separators into one joint rule
    r = np.linspace(-0.1,1.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    
    # transform data points into basis features
    h = np.concatenate((s,t),1)
    if raw_or_kernel == 0:
        h = my_utilities.create_features(h,D,feat_type)
    else:
        h = my_kernel_utilities.create_features(x,h,D,feat_type)

    f = np.dot(h,W)
    z = (np.sign(f) + 1)
    if num_classes > 2:
        z = np.argmax(f,0)+1

    ## produce rule surface
    f.shape = (np.size(f),1)
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))
    if num_classes == 2:
        ax2.contour(s,t,z,colors = 'k',linewidths = 2.5)
        ax2.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))
        ax3.plot_surface(s,t,z-1,cmap = 'gray',alpha = 0.05)
        ax3.view_init(25,-60)
    else:
        ax3.contour(s,t,z,colors = 'k',linewidths = 2.5)
        ax3.contourf(s,t,z,colors = color_opts[:],alpha = 0.1,levels = range(0,num_classes+1))   
    