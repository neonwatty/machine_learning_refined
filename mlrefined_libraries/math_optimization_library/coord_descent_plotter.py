# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Simple plotter for 3d function - from surface and contour perspective
    '''             

    # my cost history plotter
    def plot_cost(self,g,w_history):
        # make a figure
        fig,ax= plt.subplots(1,1,figsize = (8,3))

        # compute cost vales
        cost_vals = [g(w) for w in w_history]

        # plot the cost values
        ax.plot(cost_vals)

        # cleanup graph and label axes
        ax.set_xlabel('num of (outer loop) iterations')
        ax.set_ylabel('cost function value')
        ax.set_title('boosting descent')
        
    # my cost history plotter
    def plot_coord_descent_cost_history(self,g,w_history):
        # make a figure
        fig,ax= plt.subplots(1,1,figsize = (8,3))

        # compute cost vales
        cost_vals = [g(w) for w in w_history]

        # plot the cost values
        ax.plot(cost_vals)

        # cleanup graph and label axes
        ax.set_xlabel('num of (outer loop) iterations')
        ax.set_ylabel('cost function value')

        tickrange =  np.arange(0,len(w_history),len(w_history[-1]))
        tickrange2 = [int(v/float(len(w_history[-1]))) for v in tickrange]
        if len(tickrange2) > 10:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            tickrange = f(10,len(w_history))
            tickrange2 = [int(v/float(len(w_history[-1]))) for v in tickrange]
        ax.set_xticks(tickrange) # choose which x locations to have ticks
        ax.set_xticklabels(tickrange2) # set the labels to display at those ticks
    
    # show contour plot of input function
    def draw_setup(self,g,**kwargs):
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
        
    # function to plot contour of input function g and path on contours (only works for N = 2 dim input functions)
    def show_path(self,g,w_history,**kwargs):
        self.g = g
        self.w_hist = w_history
        
        ### user args
        # number of contours to show in contour plot
        self.num_contours = 15
        if 'num_contours' in kwargs:
            self.num_contours = kwargs['num_contours']
       
        ### setup figure and plot
        fig, axs = plt.subplots(1, 3, figsize=(9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,1]) 
        ax = plt.subplot(gs[1],aspect = 'equal'); 
        ax2 = plt.subplot(gs[0]); ax2.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');
        
        ### draw contours
        self.draw_contour_plot(ax,fig)
        
        ### draw path on contours
        self.draw_weight_path(ax)
            
        ### cleanup panel
        ax.set_xlabel('$w_1$',fontsize = 12)
        ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        
    ### function for drawing weight history path
    def draw_weight_path(self,ax):
        # make color range for path
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

        ### plot function decrease plot in right panel
        for j in range(len(self.w_hist)):  
            w_val = self.w_hist[j]
            g_val = self.g(w_val)

            # plot each weight set as a point
            ax.scatter(w_val[0],w_val[1],s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 2*math.sqrt((1/(float(j) + 1))),zorder = 3)

            # plot connector between points for visualization purposes
            if j > 0:
                w_old = self.w_hist[j-1]
                w_new = self.w_hist[j]
                g_old = self.g(w_old)
                g_new = self.g(w_new)
         
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 2 + 0.4,alpha = 1,zorder = 1)      # plot approx
             
    ### function for creating contour plot
    def draw_contour_plot(self,ax,fig):
        # set viewing limits on contour plot
        xvals = [self.w_hist[s][0] for s in range(len(self.w_hist))]
        yvals = [self.w_hist[s][1] for s in range(len(self.w_hist))]
        xmax = copy.deepcopy(max(xvals))
        xmin = copy.deepcopy(min(xvals))
        xgap = (xmax - xmin)*0.5
        ymax = copy.deepcopy(max(yvals))
        ymin = copy.deepcopy(min(yvals))
        ygap = (ymax - ymin)*0.5
        xmin -= xgap
        xmax += xgap
        ymin -= ygap
        ymax += ygap
            
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,400)
        w2 = np.linspace(ymin,ymax,400)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ self.g(np.reshape(s,(2,1))) for s in h])

        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cutoff = 0.5
        cutoff = (levelmax - levelmin)*cutoff
        numper = 3
        levels1 = np.linspace(cutoff,levelmax,numper)
        self.num_contours -= numper

        levels2 = np.linspace(levelmin,cutoff,min(self.num_contours,numper))
        levels = np.unique(np.append(levels1,levels2))
        self.num_contours -= numper
        while self.num_contours > 0:
            cutoff = levels[1]
            levels2 = np.linspace(levelmin,cutoff,min(self.num_contours,numper))
            levels = np.unique(np.append(levels2,levels))
            self.num_contours -= numper

        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        b = ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')