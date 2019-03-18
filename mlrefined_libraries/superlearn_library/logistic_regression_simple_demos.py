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


class visualizer:
    '''
    Visualize linear regression applied to a 2-class dataset.
    '''
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

    def center_data(self):
        # center data
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        
    def run_algo(self,algo,**kwargs):
        # Get function and compute gradient
        self.g = self.linear_least_squares
        self.grad = compute_grad(self.g)
        
        # choose algorithm
        self.algo = algo
        if self.algo == 'gradient_descent':
            self.alpha = 10**-3
            if 'alpha' in kwargs:
                self.alpha = kwargs['alpha']
        
        self.max_its = 10
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
            
        self.w_init = np.random.randn(2)
        if 'w_init' in kwargs:
            self.w_init = kwargs['w_init']
            self.w_init = np.asarray([float(s) for s in self.w_init])
            self.w_init.shape = (np.size(self.w_init),1)
            
        # run algorithm of choice
        if self.algo == 'gradient_descent':
            self.w_hist = []
            self.gradient_descent()
        if self.algo == 'newtons_method':
            self.hess = compute_hess(self.g)           # hessian of input function
            self.beta = 0
            if 'beta' in kwargs:
                self.beta = kwargs['beta']
            self.w_hist = []
            self.newtons_method()  
    
    ######## linear regression functions ########    
    def linear_least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            cost +=(w[0] + np.dot(w[1:].T,x_p) - y_p)**2
        return cost
    
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            cost +=(np.sign(w[0] + np.dot(w[1:].T,x_p)) - y_p)**2
        return cost
    
    # run gradient descent
    def gradient_descent(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        for k in range(self.max_its):   
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            grad_eval.shape = (len(w),1)
            
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
            grad_eval /= grad_norm
            
            # decide on alpha
            alpha = self.alpha
            if self.alpha == 'backtracking':
                alpha = self.backtracking(w,grad_val)
            
            # take newtons step
            w = w - alpha*grad_eval
            
            # record
            self.w_hist.append(w)     
    
    #### run newton's method ####
    def newtons_method(self):
        w = self.w_init
        self.w_hist = []
        self.w_hist.append(w)
        for k in range(self.max_its):
            # plug in value into func and derivative
            grad_eval = self.grad(w)
            hess_eval = self.hess(w)
            
            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

            # solve linear system for weights
            w = w - np.dot(np.linalg.pinv(hess_eval + self.beta*np.eye(np.size(w))),grad_eval)
                                
            # record
            self.w_hist.append(w)   
            
    ### demo 1 - fitting a line to a step dataset, then taking sign of this line ###
    def naive_fitting_demo(self,**kwargs):
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        
        # create subplot with 2 panels
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],aspect = 'equal');
    
        #### plot data in both panels ####
        self.scatter_pts(ax1)
        self.scatter_pts(ax2)
        
        #### fit line to data and plot ####
        # make plotting range
        xmin = copy.deepcopy(min(self.x))
        xmax = copy.deepcopy(max(self.x))
        xgap = (xmax - xmin)*0.4
        xmin-=xgap
        xmax+=xgap
        
        # produce fit
        x_fit = np.linspace(xmin,xmax,300)    
        w = self.w_hist[-1]
        y_fit = w[0] + x_fit*w[1]
        
        # plot linear fit
        ax2.plot(x_fit,y_fit,color = 'lime',linewidth = 1.5) 
        
        # plot sign version of linear fit
        f = np.sign(y_fit)
        bot_ind = np.argwhere(f == -1)
        bot_ind = [s[0] for s in bot_ind]
        bot_in = x_fit[bot_ind]
        bot_out = f[bot_ind]
        ax2.plot(bot_in,bot_out,color = 'r',linewidth = 1.5,linestyle = '--') 

        top_ind = np.argwhere(f == +1)
        top_ind = [s[0] for s in top_ind]
        top_in = x_fit[top_ind]
        top_out = f[top_ind]
        ax2.plot(top_in,top_out,color = 'r',linewidth = 1.5,linestyle = '--') 
        
    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.4
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.4
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
            # label axes
            ax.set_xlabel(r'$x$', fontsize = 12)
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12)
            
            ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
            ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
            
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

            # initialize points
            ax.scatter(self.x[:,0],self.x[:,1],self.y,s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1) +1, round(xmax1), 1.0))
            ax.set_yticks(np.arange(round(xmin2) +1, round(xmax2), 1.0))

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
                   
    ###### function plotting functions #######
    def plot_cost(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(8,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0],aspect = 'equal'); 
        ax2 = plt.subplot(gs[1],projection='3d'); 
        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']
        cost = 'counting_cost'
        if 'cost' in kwargs:
            cost = kwargs['cost']
        
        # make contour plot in left panel
        self.contour_plot(ax1,viewmax,num_contours,cost)
        
        if cost == 'counting_cost':
            self.counting_cost_surface(ax2,viewmax)
            
        # make contour plot in right panel
        #self.surface_plot(ax2,viewmax,view,cost)
        
        plt.show()
        
    # plot counting cost
    def plot_counting_cost(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(8,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3,width_ratios=[1,3,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off'); 
        ax = plt.subplot(gs[1],projection='3d'); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off'); 

        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        
        # define coordinate system
        r = np.linspace(-viewmax,viewmax,300)    
        s,t = np.meshgrid(r,r)
        s.shape = (np.prod(np.shape(s)),1)
        t.shape = (np.prod(np.shape(t)),1)
        w_ = np.concatenate((s,t),axis=1)

        # define cost surface
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(self.counting_cost(w_[i,:]))
        g_vals = np.asarray(g_vals)
            
        # loop over levels and print
        s.shape = (len(r),len(r))
        t.shape = (len(r),len(r))

        levels = np.unique(g_vals)
        for u in levels:
            # make copy of cost and nan out all non level entries
            z = g_vals.copy()
            ind = np.argwhere(z != u)
            ind = [v[0] for v in ind]
            z[ind] = np.nan

            # plot the current level
            z.shape = (len(r),len(r)) 
            ax.plot_surface(s,t,z,alpha = 0.4,color = '#696969',zorder = 0,shade = True,linewidth=0)

        # set viewing angle
        ax.view_init(5,126)

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

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)
        
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,ax,wmax,view,cost):
        ##### Produce cost function surface #####
        wmax += wmax*0.1
        r = np.linspace(-wmax,wmax,200)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        if cost == 'counting_cost':
            for i in range(len(r)**2):
                g_vals.append(self.counting_cost(w_[i,:]))
            g_vals = np.asarray(g_vals)
        
        '''
        for i in range(len(r)**2):
            g_vals.append(self.least_squares(w_[i,:]))
        g_vals = np.asarray(g_vals)
        '''

        # reshape and plot the surface, as well as where the zero-plane is
        w1_vals.shape = (np.size(r),np.size(r))
        w2_vals.shape = (np.size(r),np.size(r))
        g_vals.shape = (np.size(r),np.size(r))
        
        # plot cost surface
        ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
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

        ax.set_xlabel(r'$w_0$',fontsize = 12)
        ax.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.view_init(view[0],view[1])
        
    ### visualize contour plot of cost function ###
    def contour_plot(self,ax,wmax,num_contours,cost):
        #### define input space for function and evaluate ####
        w1 = np.linspace(-wmax,wmax,300)
        w2 = np.linspace(-wmax,wmax,300)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = []
        func_vals = np.asarray([self.least_squares(s) for s in h])
            
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
                
        # clean up panel
        ax.set_xlabel('$w_0$',fontsize = 12)
        ax.set_ylabel('$w_1$',fontsize = 12,rotation = 0)
        ax.set_title(r'$g\left(w_0,w_1\right)$',fontsize = 13)

        ax.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        ax.set_xlim([-wmax,wmax])
        ax.set_ylim([-wmax,wmax])        