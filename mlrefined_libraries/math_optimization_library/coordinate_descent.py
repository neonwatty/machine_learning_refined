# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time
import copy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

class visualizer:
    '''
    Illustrates how brute force coordinate descent w/linesearch works
    ''' 

    # run coordinate descent - in this case for the demo we use coordinate gradient descent with exact line search
    def run_coordinate_descent(self):
        w = copy.deepcopy(self.w_init)
        self.w_hist = []
        self.w_hist.append(copy.deepcopy(w))
        j = 0
        for j in range(int(self.max_its)):            
            # plug in value into func and derivative
            grad_eval = self.grad(w)
                    
            # loop over coordinates
            for k in range(len(w)):                
                # strip gradient of k^th coordinate
                coord_grad = copy.deepcopy(grad_eval[k])
                                                     
                # normalize direction or no?
                if self.version == 'normalized':
                    # normaize direction, if everything is perfectly zero then move in random direction
                    grad_norm = np.linalg.norm(coord_grad)
                    if grad_norm == 0:
                        coord_grad = np.sign(2*np.random.rand(1) - 1)[0]
                    else:
                        coord_grad = np.sign(coord_grad)
                 
                ### check what sort of steplength rule to employ ###
                alpha = 0
                grad_temp = copy.deepcopy(grad_eval)
                grad_temp[k] = coord_grad
                alpha = self.exact(w,grad_temp)
                
                # take coordinate descent step - update single weight
                w[k] -= alpha*coord_grad

                # record each coordinate descent step for visualization
                self.w_hist.append(copy.deepcopy(w))            
                
    # exact linesearch module for brute coordinate descent
    def exact(self,w,grad_eval):
        # set parameters of linesearch at each step
        valmax = 10
        num_evals = 3000
        
        # set alpha range
        alpha_range = np.linspace(0,valmax,num_evals)
        
        # evaluate function over direction and alpha range, grab alpha giving lowest eval
        steps = [(w - alpha*grad_eval) for alpha in alpha_range]
        func_evals = np.array([self.g(s) for s in steps])
        ind = np.argmin(func_evals)
        best_alpha = alpha_range[ind]
        
        return best_alpha
    
    # visualize descent on multi-input function
    def run(self,g,w_init,max_its,**kwargs):        
        ### input arguments ###        
        self.g = g
        self.max_its = max_its
        self.grad = compute_grad(self.g)              # gradient of input function

        pts = 'off'
        if 'pts' in kwargs:
            pts = 'off'
            
        linewidth = 2.5
        if 'linewidth' in kwargs:
            linewidth = kwargs['linewidth']
            
        view = [20,-50]
        if 'view' in kwargs:
            view = kwargs['view']

        axes = False
        if 'axes' in kwargs:
            axes = kwargs['axes']

        plot_final = False
        if 'plot_final' in kwargs:
            plot_final = kwargs['plot_final']

        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']

        # version of gradient descent to use (normalized or unnormalized)
        self.version = 'unnormalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
            
        # get initial point 
        self.w_init = np.asarray([float(s) for s in w_init])
            
        # take in user defined maximum number of iterations
        self.max_its = max_its

        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(9,4))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
        ax = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1]) #  ,sharey = ax); 

        #### run local random search algorithm ####
        self.w_hist = []
        self.steplength = 'exact'
        self.run_coordinate_descent()
        
        # set viewing limits on contour plot
        xvals = [self.w_hist[s][0] for s in range(len(self.w_hist))]
        xvals.append(self.w_init[0])
        yvals = [self.w_hist[s][1] for s in range(len(self.w_hist))]
        yvals.append(self.w_init[1])
        xmax = max(xvals)
        xmin = min(xvals)
        xgap = (xmax - xmin)*0.1
        ymax = max(yvals)
        ymin = min(yvals)
        ygap = (ymax - ymin)*0.1
        xmin -= xgap
        xmax += xgap
        ymin -= ygap
        ymax += ygap
            
        if 'xmin' in kwargs:
            xmin = kwargs['xmin']
        if 'xmax' in kwargs:
            xmax = kwargs['xmax']
        if 'ymin' in kwargs:
            ymin = kwargs['ymin']
        if 'ymax' in kwargs:
            ymax = kwargs['ymax']  
            
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,400)
        w2 = np.linspace(ymin,ymax,400)
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
        num_contours = kwargs['num_contours']
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

        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')

        # label contour lines?
        #ax.clabel(a, inline=1, fontsize=10)

        if axes == True:
            ax.axhline(linestyle = '--', color = 'k',linewidth = 1)
            ax.axvline(linestyle = '--', color = 'k',linewidth = 1)

        # colors for points
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

            # plot in left panel
            if pts == 'True':
                ax.scatter(w_val[0],w_val[1],s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 1.5*math.sqrt((1/(float(j) + 1))),zorder = 3)

                ax2.scatter(j,g_val,s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 0.7,zorder = 3)            # plot point of tangency

            # plot connector between points for visualization purposes
            if j > 0:
                w_old = self.w_hist[j-1]
                w_new = self.w_hist[j]
                g_old = self.g(w_old)
                g_new = self.g(w_new)
         
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = colorspec[j],linewidth = linewidth,alpha = 1,zorder = 2)      # plot approx
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = linewidth + 0.4,alpha = 1,zorder = 1)      # plot approx
                ax2.plot([j-1,j],[g_old,g_new],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                ax2.plot([j-1,j],[g_old,g_new],color = 'k',linewidth = 2.5,alpha = 1,zorder = 1)      # plot approx
               
        # clean panels
        title = self.steplength
        if type(self.steplength) == float:
            title = r'$\alpha = $' + str(self.steplength)
        #ax.set_title(title,fontsize = 12)
        ax.set_xlabel('$w_1$',fontsize = 12)
        ax.set_ylabel('$w_2$',fontsize = 12,rotation = 0)
        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax2.set_xlabel('iteration',fontsize = 12)
        ax2.set_ylabel(r'$g(w)$',fontsize = 12,rotation = 0,labelpad = 25)
               
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
            
        ax.set(aspect = 'equal')
        a = ax.get_position()
        yr = ax.get_position().y1 - ax.get_position().y0
        xr = ax.get_position().x1 - ax.get_position().x0
        aspectratio=1.25*xr/yr# + min(xr,yr)
        ratio_default=(ax2.get_xlim()[1]-ax2.get_xlim()[0])/(ax2.get_ylim()[1]-ax2.get_ylim()[0])
        ax2.set_aspect(ratio_default*aspectratio)
            
        # plot
        plt.show()
   