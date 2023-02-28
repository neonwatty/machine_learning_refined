# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
from matplotlib import gridspec
import copy
from inspect import signature
from matplotlib.ticker import FormatStrFormatter

class Visualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1].T
        self.y = data[:,-1:] 
 
    ###### plot plotting functions ######
    def plot_data(self):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        # plot 2d points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 16)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
            
    # plot regression fits
    def plot_fit(self,w,model,**kwargs):        
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
  
        # scatter points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 16)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
        
        # create fit
        s = np.linspace(xmin,xmax,300)[np.newaxis,:]
        colors = ['k','magenta']
        if 'colors' in kwargs:
            colors = kwargs['colors']
        c = 0
 
        normalizer = lambda a: a
        if 'normalizer' in kwargs:
            normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        ax.plot(s.T,t.T,linewidth = 4,c = 'k',zorder = 0)
        ax.plot(s.T,t.T,linewidth = 2,c = 'lime',zorder = 0)
         
    # plot regression fits
    def plot_fit_and_feature_space(self,w,model,feat,**kwargs):        
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1]); 
        
        view = [20,20]
        if 'view' in kwargs:
            view = kwargs['view']

        ##### plot left panel in original space ####
        # scatter points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax1)

        # clean up panel
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])

        # label axes
        ax1.set_xlabel(r'$x$', fontsize = 16)
        ax1.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 10)
        
        # create fit
        s = np.linspace(xmin,xmax,300)[np.newaxis,:]
 
        normalizer = lambda a: a
        if 'normalizer' in kwargs:
            normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        
        ax1.plot(s.flatten(),t.flatten(),linewidth = 4,c = 'k',zorder = 0)    
        ax1.plot(s.flatten(),t.flatten(),linewidth = 2,c = 'lime',zorder = 0)

        #### plot fit in transformed feature space #####
        # check if feature transform has internal parameters
        x_transformed = 0
        sig = signature(feat)
        if len(sig.parameters) == 2:
            if np.shape(w)[1] == 1:
                x_transformed = feat(normalizer(self.x),w)
            else:
                x_transformed = feat(normalizer(self.x),w[0])
        else: 
            x_transformed = feat(normalizer(self.x))
        
        # two dimensional transformed feature space
        if x_transformed.shape[0] == 1:
            s = np.linspace(xmin,xmax,300)[np.newaxis,:]
            
            # scatter points
            xmin,xmax,ymin,ymax = self.scatter_pts_2d(x_transformed,ax2)
        
            # produce plot
            s2 = copy.deepcopy(s)
            if len(sig.parameters) == 2:
                if np.shape(w)[1] == 1:
                    s2 = feat(normalizer(s),w)
                else:
                    s2 = feat(normalizer(s),w[0])
            else: 
                s2 = feat(normalizer(s))
            t = model(normalizer(s),w)
            
            ax2.plot(s2.flatten(),t.flatten(),linewidth = 4,c = 'k',zorder = 0)    
            ax2.plot(s2.flatten(),t.flatten(),linewidth = 2,c = 'lime',zorder = 0)
            
            # label axes
            ax2.set_xlabel(r'$f\left(x,\mathbf{w}^{\star}\right)$', fontsize = 16)
            ax2.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 10)
            
        # three dimensional transformed feature space
        if x_transformed.shape[0] == 2:
            # create panel
            ax2 = plt.subplot(gs[1],projection = '3d');  
            s = np.linspace(xmin,xmax,100)[np.newaxis,:]

            # plot data in 3d
            xmin,xmax,xmin1,xmax1,ymin,ymax = self.scatter_3d_points(x_transformed,ax2)

            # create and plot fit
            s2 = copy.deepcopy(s)
            if len(sig.parameters) == 2:
                s2 = feat(normalizer(s),w[0])
            else: 
                s2 = feat(normalizer(s))
 
            # reshape for plotting
            a = s2[0,:]
            b = s2[1,:]
            a = np.linspace(xmin,xmax,100)
            b = np.linspace(xmin1,xmax1,100)
            a,b = np.meshgrid(a,b)
            
            # get firstem
            a.shape = (1,np.size(s)**2)
            f1 = feat(normalizer(a))[0,:]
            
            # secondm
            b.shape = (1,np.size(s)**2)
            f2 = feat(normalizer(b))[1,:]
            
            # tack a 1 onto the top of each input point all at once
            c = np.vstack((a,b))
            o = np.ones((1,np.shape(c)[1]))
            c = np.vstack((o,c))
            r = (np.dot(c.T,w))
            
            # various
            a.shape = (np.size(s),np.size(s))
            b.shape = (np.size(s),np.size(s))
            r.shape = (np.size(s),np.size(s))
            ax2.plot_surface(a,b,r,alpha = 0.1,color = 'lime',rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k')
            ax2.set_xlim([np.min(a),np.max(a)])
            ax2.set_ylim([np.min(b),np.max(b)])
            
            '''
            a,b = np.meshgrid(t1,t2)
            a.shape = (1,np.size(s)**2)
            b.shape = (1,np.size(s)**2)
            '''
 
            '''
            c = np.vstack((a,b))
            o = np.ones((1,np.shape(c)[1]))
            c = np.vstack((o,c))

            # tack a 1 onto the top of each input point all at once
            r = (np.dot(c.T,w))

            a.shape = (np.size(s),np.size(s))
            b.shape = (np.size(s),np.size(s))
            r.shape = (np.size(s),np.size(s))
            ax2.plot_surface(a,b,r,alpha = 0.1,color = 'lime',rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k')
            '''
            
            # label axes
            #self.move_axis_left(ax2)
            ax2.set_xlabel(r'$f_1(x)$', fontsize = 12,labelpad = 5)
            ax2.set_ylabel(r'$f_2(x)$', rotation = 0,fontsize = 12,labelpad = 5)
            ax2.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = 0)
            self.move_axis_left(ax2)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.view_init(view[0],view[1])
 
    def scatter_pts_2d(self,x,ax):
        # set plotting limits
        xmax = copy.deepcopy(np.max(x))
        xmin = copy.deepcopy(np.min(x))
        xgap = (xmax - xmin)*0.2
        xmin -= xgap
        xmax += xgap

        ymax = copy.deepcopy(np.max(self.y))
        ymin = copy.deepcopy(np.min(self.y))
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap    

        # initialize points
        ax.scatter(x.flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 60,zorder = 3)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        return xmin,xmax,ymin,ymax
    
    ### visualize the surface plot of cost function ###
    def scatter_3d_points(self,x,ax):
        # set plotting limits
        xmax = copy.deepcopy(np.max(x[0,:]))
        xmin = copy.deepcopy(np.min(x[0,:]))
        xgap = (xmax - xmin)*0.2
        xmin -= xgap
        xmax += xgap
        
        xmax1 = copy.deepcopy(np.max(x[1,:]))
        xmin1 = copy.deepcopy(np.min(x[1,:]))
        xgap1 = (xmax1 - xmin1)*0.2
        xmin1 -= xgap1
        xmax1 += xgap1

        ymax = copy.deepcopy(np.max(self.y))
        ymin = copy.deepcopy(np.min(self.y))
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap   
        
        # plot data
        ax.scatter(x[0,:].flatten(),x[1,:].flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)
            
        # clean up panel
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
      
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        
        return xmin,xmax,xmin1,xmax1,ymin,ymax

    # set axis in left panel
    def move_axis_left(self,ax):
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view) 