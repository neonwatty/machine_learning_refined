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
    def plot_three_fits(self,run1,run2,run3,**kwargs):   
        ## strip off model, normalizer, etc., ##
        model1 = run1.model
        model2 = run2.model
        model3 = run3.model
        
        all_colors = [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']

        normalizer1 = run1.normalizer
        normalizer2 = run2.normalizer
        normalizer3 = run3.normalizer

        # get weights
        cost_history1 = run1.cost_histories[0]
        ind1 = np.argmin(cost_history1)
        w1 = run1.weight_histories[0][ind1]
        cost_history2 = run2.cost_histories[0]
        ind2 = np.argmin(cost_history2)
        w2 = run2.weight_histories[0][ind2]
        cost_history3 = run3.cost_histories[0]
        ind3 = np.argmin(cost_history3)
        w3 = run3.weight_histories[0][ind3]
        
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(10,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); 

        for ax in [ax1,ax2,ax3]:
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

            # plot model
            t = 0
            if ax == ax1:
                t = model1(normalizer1(s),w1)
                ax.set_title('underfitting',fontsize = 14)
            if ax == ax2:
                t = model2(normalizer2(s),w2)
                ax.set_title('overfitting',fontsize = 14)
            if ax == ax3:
                t = model3(normalizer3(s),w3)
                ax.set_title('"just right"',fontsize = 14)

            ax.plot(s.T,t.T,linewidth = 3,c = 'k')
            ax.plot(s.T,t.T,linewidth = 2.5,c = all_colors[2])
 
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
        ax.scatter(x.flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 60)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        return xmin,xmax,ymin,ymax