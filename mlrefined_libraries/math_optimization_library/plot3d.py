# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
import autograd.numpy as np
import math
import time

class visualizer:
    '''
    Simple plotter for 3d function - from surface and contour perspective
    '''             

    def draw_2d(self,g,**kwargs):
        self.g = g                         # input function        
        wmin = -3.1
        wmax = 3.1
        view = [50,50]
        num_contours = 10
        if 'wmin' in kwargs:            
            wmin = kwargs['wmin']
        if 'wmax' in kwargs:
            wmax = kwargs['wmax']
        if 'view' in kwargs:
            view = kwargs['view']
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']   
            
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (9,3))

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,2]) 
        ax = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1],aspect='equal'); 

        #### define input space for function and evaluate ####
        w = np.linspace(-wmax,wmax,200)
        w1_vals, w2_vals = np.meshgrid(w,w)
        w1_vals.shape = (len(w)**2,1)
        w2_vals.shape = (len(w)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([g(s) for s in h])
        w1_vals.shape = (len(w),len(w))
        w2_vals.shape = (len(w),len(w))
        func_vals.shape = (len(w),len(w))

        ### plot function as surface ### 
        ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

        # plot z=0 plane 
        ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 
        
        ### plot function as contours ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 0.5
        cutoff = (levelmax - levelmin)*cutoff
        numper = 3
        levels1 = np.linspace(cutoff,levelmax,numper)
        num_contours -= numper

        levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        num_contours -= numper
        while num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            num_contours -= numper

        ax2.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        ax2.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
        
        ### cleanup panels ###
        ax.set_xlabel('$w_1$',fontsize = 12)
        ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        ax.set_title('$g(w_1,w_2)$',fontsize = 12)
        ax.view_init(view[0],view[1])

        ax2.set_xlabel('$w_1$',fontsize = 12)
        ax2.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.set_xticks(np.arange(-round(wmax),round(wmax)+1))
        ax2.set_yticks(np.arange(-round(wmax),round(wmax)+1))

        # clean up axis
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plot
        plt.show()