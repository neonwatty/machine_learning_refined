import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd import hessian as compute_hess
from autograd import grad as compute_grad
from autograd.misc.flatten import flatten_func

import math
import time
import copy


### gradient descent ###
def normalized_gradient_descent(g,w,**kwargs):                
    
    grad = compute_grad(g)
    
    # parse optional arguments        
    max_its = 100
    if 'max_its' in kwargs:
        max_its = kwargs['max_its']
    alpha = 10**-4
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
   
    # create container for weight history 
    w_hist = []
    w_hist.append(w)
    
    # start gradient descent loop
    if verbose == True:
        print ('starting optimization...')
    for k in range(max_its):   
        # plug in value into func and derivative
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)
        
        grad_norm = np.linalg.norm(grad_eval)
        if grad_norm == 0:
            grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
        grad_eval /= grad_norm
            
        ### take gradient descent step ###
        w = w - alpha*grad_eval
        
        # record
        w_hist.append(w)     
     
    if verbose == True:
        print ('...optimization complete!')
        time.sleep(1.5)
        clear_output()
    
    return w_hist




# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history,cost_history




class entropy_visualizer:
    '''
    Visualize logistic regression applied to a 2-class dataset with N = 2
    '''
    #### initialize ####
    def __init__(self,data,g):
        # grab input
        data = data.T
        self.data = data
        self.x = data[:,:-1]
        self.y = data[:,-1]
        self.g = g
        
        # colors for viewing classification data 'from above'
        self.colors = ['cornflowerblue','salmon','lime','bisque','mediumaquamarine','b','m','g']
    
    ### logistic functionality ###
    def identity(self,t):
        val = 0
        if t > 0.5:
            val = 1
        return val
    
    # define sigmoid function
    def sigmoid(self,t):
        return 1/(1 + np.exp(-t))
    
    ######## 2d functions ########
    # animate gradient descent or newton's method
    def animate_run(self,savepath,w_hist,**kwargs):     
        self.w_hist = w_hist
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,3))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = copy.deepcopy(min(self.x))
        xmax = copy.deepcopy(max(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']        
        self.contour_plot(ax2,viewmax,num_contours)
        
        # start animation
        num_frames = len(self.w_hist)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # current color
            color = self.colorspec[k]

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            ###### make left panel - plot data and fit ######
            # initialize fit
            w = self.w_hist[k]
            y_fit = self.sigmoid(w[0] + x_fit*w[1])
            
            # scatter data
            self.scatter_pts(ax1)
            
            # plot fit to data
            ax1.plot(x_fit,y_fit,color = color,linewidth = 2) 

            ###### make right panel - plot contour and steps ######
            if k == 0:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
            if k > 0 and k < num_frames:
                self.plot_pts_on_contour(ax2,k,color)
            if k == num_frames -1:
                ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
               
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()
            
    def sigmoid(self,t):
        return 1/(1 + np.exp(-t))
    
    # produce static image of gradient descent or newton's method run
    def static_fig(self,w_hist,**kwargs):
        self.w_hist = w_hist
        ind = -1
        show_path = True
        if np.size(w_hist) == 0:
            show_path = False
        w = 0
        if show_path:
            w = w_hist[ind]
        
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,3))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # produce color scheme
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        self.colorspec = []
        self.colorspec = np.concatenate((s,np.flipud(s)),1)
        self.colorspec = np.concatenate((self.colorspec,np.zeros((len(s),1))),1)
        
        # seed left panel plotting range
        xmin = copy.deepcopy(min(self.x))
        xmax = copy.deepcopy(max(self.x))
        xgap = (xmax - xmin)*0.1
        xmin-=xgap
        xmax+=xgap
        x_fit = np.linspace(xmin,xmax,300)
        
        # seed right panel contour plot
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        num_contours = 15
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']   
            
        ### contour plot in right panel ###
        self.contour_plot(ax2,viewmax,num_contours)
        
        ### make left panel - plot data and fit ###
        # scatter data
        self.scatter_pts(ax1)
        
        if show_path:
            # initialize fit
            y_fit = self.sigmoid(w[0] + x_fit*w[1])

            # plot fit to data
            color = self.colorspec[-1]
            ax1.plot(x_fit,y_fit,color = color,linewidth = 2) 

            # add points to right panel contour plot
            num_frames = len(self.w_hist)
            for k in range(num_frames):
                # current color
                color = self.colorspec[k]

                # current weights
                w = self.w_hist[k]

                ###### make right panel - plot contour and steps ######
                if k == 0:
                    ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
                if k > 0 and k < num_frames:
                    self.plot_pts_on_contour(ax2,k,color)
                if k == num_frames -1:
                    ax2.scatter(w[0],w[1],s = 90,facecolor = color,edgecolor = 'k',linewidth = 0.5, zorder = 3)
        
        plt.show()
            
    
    ###### plot plotting functions ######
    def plot_data(self,**kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,3))

        if np.shape(self.x)[1] == 1:
            # create subplot with 2 panels
            gs = gridspec.GridSpec(1, 3, width_ratios=[1,2,1]) 
            ax1 = plt.subplot(gs[0]); ax1.axis('off') 
            ax2 = plt.subplot(gs[1]); 
            ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
            # scatter points
            self.scatter_pts(ax2)
            
        if np.shape(self.x)[1] == 2:
            gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
            ax1 = plt.subplot(gs[0],projection='3d'); 
            ax2 = plt.subplot(gs[1],aspect = 'equal'); 
            #gs.update(wspace=0.025, hspace=0.05) # set spacing between axes. 

            
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
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
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
            ax.set_title('data', fontsize = 13)
            
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
            plt.scatter(self.data[inds,0],self.data[inds,1],color = self.colors[int(count)],linewidth = 1,marker = 'o',edgecolor = 'k',s = 50)
            count+=1
            
        # clean up panel
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])

        ax.set_xticks(np.arange(round(xmin1), round(xmax1) + 1, 1.0))
        ax.set_yticks(np.arange(round(xmin2), round(xmax2) + 1, 1.0))

        # label axes
        ax.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 0)
        ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            
    # plot points on contour
    def plot_pts_on_contour(self,ax,j,color):
        # plot connector between points for visualization purposes
        w_old = self.w_hist[j-1]
        w_new = self.w_hist[j]
        g_old = self.g(w_old)
        g_new = self.g(w_new)
     
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = color,linewidth = 3,alpha = 1,zorder = 2)      # plot approx
        ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 3 + 1,alpha = 1,zorder = 1)      # plot approx
    
    ###### function plotting functions #######
    def plot_ls_cost(self,**kwargs):
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
        
        # make contour plot in left panel
        self.contour_plot(ax1,viewmax,num_contours)
        
        # make contour plot in right panel
        self.surface_plot(ax2,viewmax,view)
        
        plt.show()
        
    ### visualize the surface plot of cost function ###
    def surface_plot(self,ax,wmax,view):
        ##### Produce cost function surface #####
        wmax += wmax*0.1
        r = np.linspace(-wmax,wmax,200)

        # create grid from plotting range
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        w_ = np.concatenate((w1_vals,w2_vals),axis = 1)
        g_vals = []
        for i in range(len(r)**2):
            g_vals.append(self.g(w_[i,:]))
        g_vals = np.asarray(g_vals)

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
    def contour_plot(self,ax,wmax,num_contours):
        
        #### define input space for function and evaluate ####
        w1 = np.linspace(-wmax,wmax,100)
        w2 = np.linspace(-wmax,wmax,100)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ self.g(np.reshape(s,(2,1))) for s in h])

        #func_vals = np.asarray([self.g(s) for s in h])
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





class logistic_regression_visualizer:
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
    
    def identity(self,t):
        val = 0
        if t > 0.5:
            val = 1
        return val
        
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            # strip of pth point
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            
            # produce predicted label
            val_p = self.identity(w[0] + np.dot(w[1:].T,x_p)) 

            # compare with true label
            if val_p != y_p:
                cost += 1
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
        f = np.array([self.identity(v) for v in y_fit])
        bot_ind = np.argwhere(f == 0)
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
            ax.set_ylabel(r'$y$', rotation = 0,fontsize = 12,labelpad = 10)
            
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


class cost_visualizer:
    '''
    Visualize an input cost function based on data.
    '''
    
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]
    
    def identity(self,t):
        val = 0
        if t > 0.5:
            val = 1
        return val
        
    def counting_cost(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            # strip of pth point
            x_p = copy.deepcopy(self.x[p,:])
            x_p.shape = (len(x_p),1)
            y_p = self.y[p]
            
            # produce predicted label
            val_p = self.identity(w[0] + np.dot(w[1:].T,x_p)) 

            # compare with true label
            if val_p != y_p:
                cost += 1
        return float(cost)
    
    # sigmoid
    def sigmoid(self,t):
        return 1/(1 + np.exp(-t))
    
    def my_exp(self,t):
        if t > 10:
            t = 10
        if t < -10:
            t = -10
        return np.exp(t)
    
    # log-loss
    def entropy(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            # extract point
            x_p = self.x[p,:]
            y_p = self.y[p]
            
            # sigmoid of linear combo of input
            a_p = self.sigmoid(w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])]))
            
            # compute cost
            cost -= (y_p*np.log(a_p) + (1-y_p)*np.log(1 - a_p))
        return cost
    
    # sigmoid non-convex least squares
    def sigmoid_least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p,:]
            y_p = self.y[p]
            a_p = w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])])
            cost +=(self.sigmoid(a_p) - y_p)**2
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
        g = self.sigmoid_least_squares
        self.surface_plot(g,ax2,viewmax,view)
        g = self.entropy
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



class LS_visualizer:
    '''
    Visualize an input cost function based on data.
    '''
    
    #### initialize ####
    def __init__(self,data):
        # grab input
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]
    
    # sigmoid
    def sigmoid(self,t):
        return 1/(1 + np.exp(-t))
    
    # sigmoid non-convex least squares
    def sigmoid_least_squares(self,w):
        cost = 0
        for p in range(0,len(self.y)):
            x_p = self.x[p,:]
            y_p = self.y[p]
            a_p = w[0] + np.sum([u*v for (u,v) in zip(x_p,w[1:])])
            cost +=(self.sigmoid(a_p) - y_p)**2
        return cost

    ###### function plotting functions #######
    def plot_costs(self,**kwargs):    
        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(6,3))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[.75,1]) 
        ax1 = plt.subplot(gs[0]);
        self.scatter_pts(ax1)
        ax2 = plt.subplot(gs[1],projection='3d'); 

        
        # pull user-defined args
        viewmax = 3
        if 'viewmax' in kwargs:
            viewmax = kwargs['viewmax']
        view = [20,100]
        if 'view' in kwargs:
            view = kwargs['view']
        label_axes = True
        if 'label_axes' in kwargs:
            label_axes = kwargs['label_axes']
        
        # make contour plot in each panel
        g = self.sigmoid_least_squares
        self.surface_plot(g,ax2,viewmax,view)
        
        if label_axes == True:
            ax2.set_xlabel(r'$w_0$',fontsize = 12)
            ax2.set_ylabel(r'$w_1$',fontsize = 12,rotation = 0)
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
                ax.plot_surface(w1_vals,w2_vals,z,alpha = 1,color = '#696969',zorder = 0,shade = True,linewidth=0)

        else: # smooth cost function, plot usual
            # reshape and plot the surface, as well as where the zero-plane is
            g_vals.shape = (np.size(r),np.size(r))

            # plot cost surface
            ax.plot_surface(w1_vals,w2_vals,g_vals,alpha = 1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)  
        
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

        ax.view_init(view[0],view[1])
        
        
    # scatter points
    def scatter_pts(self,ax):
        if np.shape(self.x)[1] == 1:
            # set plotting limits
            xmax = max(copy.deepcopy(self.x))
            xmin = min(copy.deepcopy(self.x))
            xgap = (xmax - xmin)*0.4
            xmin -= xgap
            xmax += xgap
            
            ymax = max(copy.deepcopy(self.y))
            ymin = min(copy.deepcopy(self.y))
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
   
          