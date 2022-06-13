# import standard plotting 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output

# other basic libraries
import math
import time
import copy
import autograd.numpy as np

# patch / convex hull libraries
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull

# import optimizer class from same library
from . import optimizers

class Visualizer:
    '''
    Demonstrate multiclass logistic regression classification
    
    '''
    
    #### initialize ####
    def __init__(self,data):    
        data = data.T
        
        # define the input and output of our dataset
        self.x = np.asarray(data[:,:-1])
        self.x.shape = (len(self.x),np.shape(data)[1]-1); self.x = self.x.T;
        self.y = data[:,-1]
        self.y.shape = (len(self.y),1)
    
        # colors for viewing classification data 'from above'
        self.colors = [[ 0, 0.4, 1],[1,0,0.4],[0, 1, 0.5],[1, 0.7, 0.5],'violet','mediumaquamarine']
        
    ###### cost functions ######
    # counting cost for multiclass classification - used to determine best weights
    def counting_cost(self,W):        
        # pre-compute predictions on all points
        y_hats = W[0,:] + np.dot(self.x.T,W[1:,:])

        # compute counting cost
        cost = 0
        for p in range(len(self.y)):
            # pluck out current true label, predicted label
            y_p = int(self.y[p][0])         # subtract off one due to python indexing
            y_hat_p = int(np.argmax(y_hats[p])) 

            # update cost
            cost += np.abs(np.sign(y_hat_p - y_p))
        return cost
    
    
    # multiclass softmaax regularized by the summed length of all normal vectors
    def multiclass_softmax(self,W):        
        # pre-compute predictions on all points
        all_evals = W[0,:] + np.dot(self.x.T,W[1:,:])

        # compute counting cost
        cost = 0
        for p in range(len(self.y)):
            # pluck out current true label
            y_p = int(self.y[p][0])    # subtract off one due to python indexing

            # update cost summand
            cost +=  np.log(np.sum(np.exp(all_evals[p,:]))) - all_evals[p,y_p]

        # return cost with regularizer added
        return cost + self.lam*np.linalg.norm(W[1:,:],'fro')**2

    
    ###### plotting functions ######  
    # show data
    def show_dataset(self):
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        gs = gridspec.GridSpec(1, 3,width_ratios = [1,3,1]) 

        # setup current axis
        ax = plt.subplot(gs[1],aspect = 'equal'); 
        
        # run axis through data plotter
        self.plot_data(ax)
        
        # determine plotting ranges
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        
        plt.show()
        
    # show coloring of entire space
    def show_complete_coloring(self,w_hist,**kwargs):
        '''
        # determine best set of weights from history
        cost_evals = []
        for i in range(len(w_hist)):
            W = w_hist[i]
            cost = self.counting_cost(W)
            cost_evals.append(cost)
        ind = np.argmin(cost_evals)
        '''
        
        # or just take last weights
        self.W = w_hist[-1]
        
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        
        show_cost = False
        if 'show_cost' in kwargs:
            show_cost = kwargs['show_cost']
        if show_cost == True:   
            gs = gridspec.GridSpec(1, 3,width_ratios = [1,1,1],height_ratios = [1]) 
            
            # create third panel for cost values
            ax3 = plt.subplot(gs[2],aspect = 'equal')
            
        else:
            gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        # generate input range for viewing range
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # plot panel with all data and separators
        self.plot_data(ax)
        self.plot_data(ax2)
        self.plot_all_separators(ax)
                
        ### draw multiclass boundary on right panel
        r = np.linspace(minx,maxx,2000)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        h = np.concatenate([o,w1_vals,w2_vals],axis = 1)
        pts = np.dot(h,self.W)
        g_vals = np.argmax(pts,axis = 1)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot contour
        C = len(np.unique(self.y))
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 2.75,zorder = 4)
        ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        ax.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))
        
        # dress panel
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #ax.set_ylabel(r'$x_2$',rotation = 0,fontsize = 12,labelpad = 10)
        #ax.set_xlabel(r'$x_1$',fontsize = 12)
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        #ax2.set_ylabel(r'$x_2$',rotation = 0,fontsize = 12,labelpad = 10)
        #ax2.set_xlabel(r'$x_1$',fontsize = 12)

        
        if  show_cost == True: 
            # compute cost eval history
            g = kwargs['cost']
            cost_evals = []
            for i in range(len(w_hist)):
                W = w_hist[i]
                cost = g(W)
                cost_evals.append(cost)
     
            # plot cost path - scale to fit inside same aspect as classification plots
            num_iterations = len(w_hist)
            s = np.linspace(minx + gapx,maxx - gapx,num_iterations)
            scaled_costs = [c/float(max(cost_evals))*(maxx-gapx) - (minx+gapx) for c in cost_evals]
            ax3.plot(s,scaled_costs,color = 'k',linewidth = 1.5)
            ax3.set_xlabel('iteration',fontsize = 12)
            ax3.set_title('cost function plot',fontsize = 12)
            
            # rescale number of iterations and cost function value to fit same aspect ratio as other two subplots
            ax3.set_xlim(minx,maxx)
            ax3.set_ylim(minx,maxx)
            
            ### set tickmarks for both axes - requries re-scaling   
            # x axis
            marks = range(0,num_iterations,round(num_iterations/5.0))
            ax3.set_xticks(s[marks])
            labels = [item.get_text() for item in ax3.get_xticklabels()]
            ax3.set_xticklabels(marks)
            
            ### y axis
            r = (max(scaled_costs) - min(scaled_costs))/5.0
            marks = [min(scaled_costs) + m*r for m in range(6)]
            ax3.set_yticks(marks)
            labels = [item.get_text() for item in ax3.get_yticklabels()]
            
            r = (max(cost_evals) - min(cost_evals))/5.0
            marks = [int(min(cost_evals) + m*r) for m in range(6)]
            ax3.set_yticklabels(marks)
   

    # show coloring of entire space
    def show_discrete_step(self,w_hist,view,**kwargs):
        '''
        # determine best set of weights from history
        cost_evals = []
        for i in range(len(w_hist)):
            W = w_hist[i]
            cost = self.counting_cost(W)
            cost_evals.append(cost)
        ind = np.argmin(cost_evals)
        '''
        
        # or just take last weights
        self.W = w_hist[-1]
        
        # initialize figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 2,width_ratios = [1.5,1]) 
            
        # setup current axis
        ax = plt.subplot(gs[1],projection = '3d');
        ax2 = plt.subplot(gs[0],aspect = 'equal');
        
        # load in args
        zplane = 'on'
        if 'zplane' in kwargs:
            zplane = kwargs['zplane']
       
        # generate input range for viewing range
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # plot panel with all data and separators
        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)

        # plot data in right panel from above
        self.plot_data(ax2)
                
        ### draw multiclass boundary on right panel
        r = np.linspace(minx,maxx,4000)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        o = np.ones((len(r)**2,1))
        h = np.concatenate([o,w1_vals,w2_vals],axis = 1)
        pts = np.dot(h,self.W)
        g_vals = np.argmax(pts,axis = 1) 

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))
        
        # plot contour in right panel
        C = len(np.unique(self.y))
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(-1,C),linewidths = 2.75,zorder = 4)
        ax2.contourf(w1_vals,w2_vals,g_vals,colors = self.colors[:],alpha = 0.2,levels = range(-1,C))
        
        ### plot discrete step function ###
        # to plot the step function, plot the bottom and top steps separately - z1 and z2
        steps = np.unique(g_vals)
        num_steps = np.arange(len(steps))
        
        # loop over each step and plot
        g_vals_copy = copy.deepcopy(g_vals)
        g_vals_copy.shape = (len(r)**2,1)
        for step in steps:
            # copy surface            
            g_copy = np.zeros((len(r)**2,1))
            g_copy.fill(np.nan)
            
            # find step in copy, nan out all else
            ind = np.argwhere(g_vals_copy == step)
            ind = [v[0] for v in ind]
            for i in ind:
                g_copy[i] = step
            
            # reshape and plot
            g_copy.shape = (len(r),len(r))
            ax.plot_surface(w1_vals,w2_vals,g_copy,alpha = 0.25,color = 'w',zorder = 0,edgecolor = 'k',linewidth=0.25,cstride = 200, rstride = 200,shade=10,antialiased=True)

        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            g_vals +=1
            #ax.plot_surface(w1_vals,w2_vals,g_vals*0-1,alpha = 0.1) 
            
            # loop over each class and color in z-plane
            for c in class_nums:
                # plot separator curve in left plot z plane
                ax.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
                        
                # color parts of plane with correct colors
                ax.contourf(w1_vals,w2_vals, g_vals - 0.5 - c ,colors = self.colors[(int(c)):],alpha = 0.1,levels = range(0,2))
                
                
        # scatter points in 3d
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            ax.scatter(self.x[0,ind],self.x[1,ind],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5,zorder = 3)
            
        # dress panel
        ax.view_init(view[0],view[1])
        #ax.axis('off')
        ax.set_xlim(minx,maxx)
        ax.set_ylim(minx,maxx)
        ax.set_zlim(-0.1,C - 1+0.1)
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xlim(minx,maxx)
        ax2.set_ylim(minx,maxx)
        #ax2.set_ylabel(r'$x_2$',rotation = 0,fontsize = 12,labelpad = 10)
        #ax2.set_xlabel(r'$x_1$',fontsize = 12)


    ### compare grad descent runs - given cost to counting cost ###
    def compare_to_counting(self,**kwargs):
        # parse args
        num_runs = 1
        if 'num_runs' in kwargs:
            num_runs = kwargs['num_runs']
        max_its = 200
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        alpha = 10**-2
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  
        steplength_rule = 'none'
        if 'steplength_rule' in kwargs:
            steplength_rule = kwargs['steplength_rule']
        version = 'unnormalized'
        if 'version' in kwargs:
            version = kwargs['version'] 
        algo = 'gradient_descent'
        if 'algo' in kwargs:
            algo = kwargs['algo']
         
        #### perform all optimizations ###
        self.lam = 10**-3  # our regularization paramter 
        if 'lam' in kwargs:
            self.lam = kwargs['lam']
            
        g = self.multiclass_softmax
        g_count = self.counting_cost
        
        # create instance of optimizers
        self.opt = optimizers.MyOptimizers()
        
        # run optimizer
        big_w_hist = []
        C = len(np.unique(self.y))
        for j in range(num_runs):
            # create random initialization
            w_init =  np.random.randn(np.shape(self.x)[0]+1,C)

            # run algo
            if algo == 'gradient_descent':# run gradient descent
                w_hist = self.opt.gradient_descent(g = g,w = w_init,version = version,max_its = max_its, alpha = alpha,steplength_rule = steplength_rule)
            elif algo == 'newtons_method':
                w_hist = self.opt.newtons_method(g = g,w = w_init,max_its = max_its)
            big_w_hist.append(w_hist)
            
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (8,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);
        
        #### start runs and plotting ####
        for j in range(num_runs):
            w_hist = big_w_hist[j]
            
            # evaluate counting cost / other cost for each weight in history, then plot
            count_evals = []
            cost_evals = []
            for k in range(len(w_hist)):
                w = w_hist[k]
                g_eval = g(w)
                cost_evals.append(g_eval)
                
                count_eval = g_count(w)
                count_evals.append(count_eval)
                
            # plot each 
            ax1.plot(np.arange(0,len(w_hist)),count_evals[:len(w_hist)],linewidth = 2)
            ax2.plot(np.arange(0,len(w_hist)),cost_evals[:len(w_hist)],linewidth = 2)
                
        #### cleanup plots ####
        # label axes
        ax1.set_xlabel('iteration',fontsize = 13)
        ax1.set_ylabel('num misclassifications',rotation = 90,fontsize = 13)
        ax1.set_title('number of misclassifications',fontsize = 14)
        ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        ax2.set_xlabel('iteration',fontsize = 13)
        ax2.set_ylabel('cost value',rotation = 90,fontsize = 13)
        title = 'Multiclass Softmax'
        ax2.set_title(title,fontsize = 14)
        ax2.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        
        plt.show()
        
    
    #### utility functions ####           
    #plot data
    def plot_data(self,ax):
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        for a in range(0,num_classes):
            t = np.argwhere(self.y == a)
            t = t[:,0]
            ax.scatter(self.x[0,t],self.x[1,t], s = 50,color = self.colors[a],edgecolor = 'k',linewidth = 1.5)
        
    # plot separators
    def plot_all_separators(self,ax):
        # determine plotting ranges
        minx = min(min(self.x[0,:]),min(self.x[1,:]))
        maxx = max(max(self.x[0,:]),max(self.x[1,:]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx
        
        # initialize figure, plot data, and dress up panels with axes labels etc.
        num_classes = np.size(np.unique(self.y))
                
        # color current class
        r = np.linspace(minx,maxx,400)
        for a in range(0,num_classes):
            # get current weights
            w = self.W[:,a]
            
            # draw subproblem separator
            z = - w[0]/w[2] - w[1]/w[2]*r
            r = np.linspace(minx,maxx,400)
            ax.plot(r,z,linewidth = 2,color = self.colors[a],zorder = 3)
            ax.plot(r,z,linewidth = 2.75,color = 'k',zorder = 2)