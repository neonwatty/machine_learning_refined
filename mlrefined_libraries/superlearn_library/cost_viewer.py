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

class Visualizer:
    '''
    Visualize an input cost function based on data.
    '''
    
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

    # least squares
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p,:]
            y_p = self.y[p]
            a_p = w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])])
            e = 0
            if np.sign(a_p) != y_p:
                cost += 1
        return float(cost)
    
    # log-loss
    def log_loss(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p,:]
            y_p = self.y[p]
            a_p = w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])])
            cost += np.log(1 + np.exp(-y_p*a_p))
        return cost
    
    # tanh non-convex least squares
    def tanh_least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p,:]
            y_p = self.y[p]
            a_p = w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])])
            cost +=(np.tanh(a_p) - y_p)**2
        return cost

    ###### function plotting functions #######
    def plot_costs(self,**kwargs):    
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(8,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
        ax1 = plt.subplot(gs[0],projection='3d');
        ax2 = plt.subplot(gs[1],projection='3d'); 
        ax3 = plt.subplot(gs[2],projection='3d');
        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # make contour plot in each panel
        g = self.counting_cost
        self.surface_plot(g,ax1,viewmax,view)
        g = self.tanh_least_squares
        self.surface_plot(g,ax2,viewmax,view)
        g = self.log_loss
        self.surface_plot(g,ax3,viewmax,view)
        plt.show()
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,g,ax,wmax,view):
        ##### Produce cost function surface #####
        r = np.linspace(-wmax,wmax,300)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(g(w_[i,:]))
        g_vals = np.asarray(g_vals)
        
        w1_vals.shape = (np.size(r),np.size(r))
        w2_vals.shape = (np.size(r),np.size(r))
        
        ### is this a counting cost?  if so re-calculate ###
        levels = np.unique(g_vals)
        if np.size(levels) < 30:
            # plot each level of the counting cost
            levels = np.unique(g_vals)
            for u in levels:
                # make copy of cost and nan out all non level entries
                z = g_vals.copy()
                ind = np.argwhere(z != u)
                ind = [v[0] for v in ind]
                z[ind] = np.nan

                # plot the current level
                z.shape = (len(r),len(r)) 
                ax.plot_surface(w1_vals,w2_vals,z,alpha = 0.4,color = '#696969',zorder = 0,shade = True,linewidth=0)

        else: # smooth cost function, plot usual
            # reshape and plot the surface, as well as where the zero-plane is
            g_vals.shape = (np.size(r),np.size(r))

            # plot cost surface
            ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
        ### clean up panel ###
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)

        ax.view_init(view[0],view[1])
   