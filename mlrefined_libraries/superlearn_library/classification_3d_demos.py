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
    Visualize classification on a 2-class dataset with N = 2
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.data = data
        self.x = data[:,:-1]
        self.y = data[:,-1]
        
        # colors for viewing classification data 'from above'
        self.colors = ['cornflowerblue','salmon','lime','bisque','mediumaquamarine','b','m','g']

    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    # the counting cost function - for determining best weights from input weight history
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p]
            y_p = self.y[p]
            a_p = w[0] + sum([a*b for a,b in zip(w[1:],x_p)])
            cost += (np.sign(a_p) - y_p)**2
        return 0.25*cost
                    
     ######## 3d static and animation functions ########
    # produce static image of gradient descent or newton's method run
    def static_fig(self,w,**kwargs):      
        # grab args
        zplane = 'on'
        if 'zplane' in kwargs:
            zplane = kwargs['zplane']
            
        cost_plot = 'off'
        if 'cost_plot' in kwargs:
            cost_plot = kwargs['cost_plot']     
         
        g = 0
        if 'g' in kwargs:
            g = kwargs['g']             
                
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,400)
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        h = np.concatenate([x1_vals,x2_vals],axis = 1)
        g_vals = np.tanh( w[0] + w[1]*x1_vals + w[2]*x2_vals )
        g_vals = np.asarray(g_vals)

        # vals for cost surface
        x1_vals.shape = (len(r),len(r))
        x2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # create figure to plot
        num_panels = 2
        fig_len = 9
        widths = [1,1]
        if cost_plot == 'on':
            num_panels = 3
            fig_len = 8
            widths = [2,2,1]
        fig, axs = plt.subplots(1, num_panels, figsize=(fig_len,4))
        gs = gridspec.GridSpec(1, num_panels, width_ratios=widths) 
        ax1 = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 
        ax3 = 0
        if cost_plot == 'on':
            ax3 = plt.subplot(gs[2],aspect = 0.5); 

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure
            
        # plot points - first in 3d, then from above
        self.scatter_pts(ax1)
        self.separator_view(ax2)
            
        # set zaxis to the left
        self.move_axis_left(ax1)
            
        # set view
        if 'view' in kwargs:
            view = kwargs['view']
            ax1.view_init(view[0],view[1])
            
        class_nums = np.unique(self.y)
        C = len(class_nums)
            
        # plot regression surface
        ax1.plot_surface(x1_vals,x2_vals,g_vals,alpha = 0.1,color = 'k',rstride=20, cstride=20,linewidth=0,edgecolor = 'k') 
            
        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            ax1.plot_surface(x1_vals,x2_vals,g_vals*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'w',edgecolor = 'k') 
            # plot separator curve in left plot
            ax1.contour(x1_vals,x2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
                
            if C == 2:
                ax1.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[1],levels = [0,1],zorder = 1,alpha = 0.1)
                ax1.contourf(x1_vals,x2_vals,g_vals+1,colors = self.colors[0],levels = [0,1],zorder = 1,alpha = 0.1)

            
        # plot separator in right plot
        ax2.contour(x1_vals,x2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            
        # plot color filled contour based on separator
        if C == 2:
            g_vals = np.sign(g_vals) + 1
            ax2.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))
        else:
            ax2.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))
     
        # plot cost function value
        if cost_plot == 'on':
            # plot cost function history
            g_hist = []
            for j in range(len(w_hist)):
                w = w_hist[j]
                g_eval = g(w)
                g_hist.append(g_eval)
                
            g_hist = np.asarray(g_hist).flatten()
            
            # plot cost function history
            ax3.plot(np.arange(len(g_hist)),g_hist,linewidth = 2)
            ax3.set_xlabel('iteration',fontsize = 13)
            ax3.set_title('cost value',fontsize = 12)
    
        plt.show()
  
    # produce static image of gradient descent or newton's method run
    def static_fig_topview(self,w,**kwargs):               
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,400)
        x1_vals,x2_vals = np.meshgrid(r,r)
        x1_vals.shape = (len(r)**2,1)
        x2_vals.shape = (len(r)**2,1)
        h = np.concatenate([x1_vals,x2_vals],axis = 1)
        g_vals = np.tanh( w[0] + w[1]*x1_vals + w[2]*x2_vals )
        g_vals = np.asarray(g_vals)

        # vals for cost surface
        x1_vals.shape = (len(r),len(r))
        x2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # create figure to plot
        ### initialize figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]);  ax1.axis('off')
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        ax3 = plt.subplot(gs[2]);  ax3.axis('off')
            
        # plot points - first in 3d, then from above
        self.separator_view(ax2)
        class_nums = np.unique(self.y)
        C = len(class_nums)
         
        # plot separator in right plot
        ax2.contour(x1_vals,x2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            
        # plot color filled contour based on separator
        if C == 2:
            g_vals = np.sign(g_vals) + 1
            ax2.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))
        else:
            ax2.contourf(x1_vals,x2_vals,g_vals,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))
  
        plt.show()
        

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
    
    ###### plot plotting functions ######
    def plot_data(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],projection='3d'); 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 

        # plot points - first in 3d, then from above
        self.scatter_pts(ax1)
        self.separator_view(ax2)

        # set zaxis to the left
        self.move_axis_left(ax1)
            
        # set view
        if 'view' in kwargs:
            view = kwargs['view']
            ax1.view_init(view[0],view[1])
        
    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(self.x[:,0]))
            xmin1 = copy.deepcopy(min(self.x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.35
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(self.x[:,0]))
            xmin2 = copy.deepcopy(min(self.x[:,0]))
            xgap2 = (xmax2 - xmin2)*0.35
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # scatter points in both panels
            class_nums = np.unique(self.y)
            C = len(class_nums)
            for c in range(C):
                ind = np.argwhere(self.y == class_nums[c])
                ind = [v[0] for v in ind]
                ax.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1) +1, round(xmax1), 1.0))
            ax.set_yticks(np.arange(round(xmin2) +1, round(xmax2), 1.0))
            ax.set_zticks([-1,0,1])
            
            # label axes
            ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)

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

    # plot data 'from above' in seperator view
    def separator_view(self,ax):
        # set plotting limits
        xmax1 = copy.deepcopy(max(self.x[:,0]))
        xmin1 = copy.deepcopy(min(self.x[:,0]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
            
        xmax2 = copy.deepcopy(max(self.x[:,0]))
        xmin2 = copy.deepcopy(min(self.x[:,0]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
            
        ymax = max(self.y)
        ymin = min(self.y)
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap    

        # scatter points
        classes = np.unique(self.y)
        count = 0
        for num in classes:
            inds = np.argwhere(self.y == num)
            inds = [s[0] for s in inds]
            ax.scatter(self.data[inds,0],self.data[inds,1],color = self.colors[int(count)],linewidth = 1,marker = 'o',edgecolor = 'k',s = 50)
            count+=1
            
        # clean up panel
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])

        ax.set_xticks(np.arange(round(xmin1), round(xmax1) + 1, 1.0))
        ax.set_yticks(np.arange(round(xmin2), round(xmax2) + 1, 1.0))

        # label axes
        ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 0)
        ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            