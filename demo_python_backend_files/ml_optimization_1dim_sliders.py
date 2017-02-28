import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from JSAnimation import IPython_display

# -*- coding: utf-8 -*-

class ml_optimization_1dim_sliders:
    
    def __init__(self):
        a = 0
        self.data = []
        self.cost_history = []
        self.x_orig = []
        
    # load in a two-dimensional dataset from csv - input should be in first column, oiutput in second column, no headers 
    def load_data(self,*args):
        # load data
        self.data = np.asarray(pd.read_csv(args[0],header = None))
   
        # center x-value of data
        self.data[:,0] = self.data[:,0] - np.mean(self.data[:,0])
        
        # center y-value of data if not performing logistic regression
        if args[-1] != 'logistic':
            self.data[:,1] = self.data[:,1] - np.mean(self.data[:,1])
        
        # transform input if needed
        self.x_orig = self.data[:,0].copy()
        if len(args) > 1:
            if args[1] == 'sin':
                self.data[:,0] = np.sin(2*np.pi*self.data[:,0])
        
    #### linear regression functions ####    
    def compute_lin_regression_cost(self,x,y,b,w):
        cost = 0
        for p in range(0,len(y)):
            cost +=(b + w*x[p] - y[p])**2
        return cost
                
    # gradient descent function
    def run_lin_regression_grad_descent(self,inits,max_its):    
        # peel off coordinates
        x = self.data[:,0]
        y = self.data[:,1]
        
        # initialize parameters - we choose this special to illustrate whats going on
        b = inits[0]    # initial intercept
        w = inits[1]      # initial slope
        P = len(y)
        
        # plot first parameters on cost surface
        cost_val = self.compute_lin_regression_cost(x,y,b,w)
        self.cost_history = []
        self.cost_history.append([b,w,cost_val])
        
        # gradient descent loop
        for k in range(1,max_its+1):   
            # compute each partial derivative - gprime_b is partial with respect to b, gprime_w the partial with respect to w            
            gprime_b = 0
            gprime_w = 0
            for p in range(0,P):
                temp = 2*(b + w*x[p] - y[p])
                gprime_b += temp
                gprime_w += temp*x[p]
            
            # set alpha via line search
            grad = np.asarray([gprime_b,gprime_w])
            grad.shape = (len(grad),1)
            alpha = self.line_search(x,y,b,w,grad,self.compute_lin_regression_cost)
            
            # take descent step in each partial derivative
            b = b - alpha*gprime_b
            w = w - alpha*gprime_w

            # compute cost function value 
            cost_val = self.compute_lin_regression_cost(x,y,b,w)
            self.cost_history.append([b,w,cost_val])   
      
    #### logistic regression functions ####
    def compute_logistic_regression_cost(self,x,y,b,w):
        cost = 0
        for p in range(0,len(y)):
            cost += np.log(1 + np.exp(-y[p]*(b + x[p]*w)))
        return cost
    
    # gradient descent function for softmax cost/logistic regression 
    def run_logistic_regression_grad_descent(self,inits,max_its):
        # peel off coordinates
        x = self.data[:,0]
        y = self.data[:,1]
        
        # initialize parameters - we choose this special to illustrate whats going on
        b = inits[0]
        w = inits[1]
        P = len(y)
        
        # plot first parameters on cost surface
        cost_val = self.compute_logistic_regression_cost(x,y,b,w)
        self.cost_history = []
        self.cost_history.append([b,w,cost_val])
        
        for k in range(max_its):
            # compute gradient
            gprime_b = 0
            gprime_w = 0
            for p in range(P):
                temp = -1/(1 + np.exp(y[p]*(b + w*x[p])))*y[p]
                gprime_b += temp
                gprime_w += temp*x[p]
            grad = np.asarray([gprime_b,gprime_w])
            grad.shape = (len(grad),1)         
            
            # compute step length via line search
            alpha = self.line_search(x,y,b,w,grad,self.compute_logistic_regression_cost)
    
            # take descent step in each partial derivative
            b = b - alpha*gprime_b
            w = w - alpha*gprime_w

            # compute cost function value 
            cost_val = self.compute_logistic_regression_cost(x,y,b,w)
            self.cost_history.append([b,w,cost_val]) 

            
    #### line search module - used for with both linear regression and logistic regression grad descent functions ####
    def line_search(self,x,y,b,w,grad,cost_fun):
        alpha = 1
        t = 0.1
        g_w = cost_fun(x,y,b,w)
        norm_w = np.linalg.norm(grad)**2
        while cost_fun(x,y,b - alpha*grad[0],w - alpha*grad[1]) > g_w - alpha*0.5*norm_w:
            alpha = t*alpha
        return alpha

    ##### plotting functions ####
    # show the net transformation using slider
    def fitting_slider(self,**args):  
        # pull out coordinates
        x_orig = self.x_orig
        x_tran = self.data[:,0]
        y = self.data[:,1]
        
        ##### precomputations #####
        # precompute fits input
        x_fit = np.linspace(np.min(x_orig)-1, np.max(x_orig)+1, 100)
        
        # precompute surface 
        xs = max([abs(v[0]) for v in self.cost_history])
        ys = max([abs(v[1]) for v in self.cost_history])
        minval = min(-xs,-ys)
        maxval = max(xs,ys)
        gap = (maxval - minval)*0.2
        r = np.linspace(minval - gap, maxval + gap)    
        s,t = np.meshgrid(r,r)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))

        # generate surface based on given data - done very lazily - recomputed each time
        g = 0
        P = len(y)
        if args['fit_type'] == 'line fit' or args['fit_type'] == 'sine fit':
            for p in range(0,P):
                g+= (s + t*x_tran[p] - y[p])**2
        if args['fit_type'] == 'logistic fit':
            for p in range(0,P):
                g+= np.log(1 + np.exp(-y[p]*(s + t*x_tran[p])))

        # reshape and plot the surface, as well as where the zero-plane is
        s.shape = (np.size(r),np.size(r))
        t.shape = (np.size(r),np.size(r))
        g.shape = (np.size(r),np.size(r))
        
        # setup figure to plot
        fig = plt.figure(num=None, figsize=(12,4), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122,projection='3d')

        # slider mechanism
        def show_fit(step):
            ### initialize plot data points and fit
            # initialize fit
            vals = self.cost_history[step]
            b = vals[0]
            w = vals[1]
            yfit = 0
            # transform input if needed for plotting
            if args['fit_type'] == 'line fit':
                y_fit = b + x_fit*w
            if args['fit_type'] == 'sine fit':
                y_fit = b + np.sin(2*np.pi*x_fit)*w
            if args['fit_type'] == 'logistic fit':
                y_fit = np.tanh(b + x_fit*w)

            # plot fit to data
            ax1.cla()
            ax1.plot(x_fit,y_fit,'-r',linewidth = 3) 

            # initialize points
            ax1.scatter(x_orig,y)

            # clean up panel
            xgap = float(max(x_orig) - min(x_orig))/float(10)
            ax1.set_xlim([min(x_orig)-xgap,max(x_orig)+xgap])
            ygap = float(max(y) - min(y))/float(10)
            ax1.set_ylim([min(y)-ygap,max(y)+ygap])
            ax1.set_xticks([])
            ax1.set_yticks([])

            ### plot surface
            ax2.cla()
            artist = ax2.plot_surface(s,t,g,alpha = 0.15)
            ax2.plot_surface(s,t,g*0,alpha = 0.1)

            # plot all gradient descent steps faintly for visualization purposes
            bs = []
            ws = []
            costs = []
            for i in range(len(self.cost_history)):
                bwg = self.cost_history[i]
                b = bwg[0]
                w = bwg[1]
                cost = bwg[2]
                bs.append(b)
                ws.append(w)
                costs.append(cost)
            ax2.scatter(bs,ws,costs,color = 'm',marker = 'x',linewidth = 3, alpha = 0.1)            

            # plot current gradient descent step in bright red
            b = vals[0]
            w = vals[1]
            cost = vals[2]
            ax2.scatter(b,w,cost,marker = 'o',color = 'r',s = 50,edgecolor = 'k',linewidth = 1)            
            
            # clean up panel
            ax2.view_init(args['view'][0],args['view'][1])        
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])

            ax2.set_xlabel(args['xlabel'],fontsize = 14,labelpad = -5)
            ax2.set_ylabel(args['ylabel'],fontsize = 14,labelpad = -5)

            ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax2.set_zlabel('cost  ',fontsize = 14, rotation = 0,labelpad = 1)
            
            return artist,
        
        anim = animation.FuncAnimation(fig, show_fit,frames=len(self.cost_history), interval=len(self.cost_history), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = int(len(self.cost_history)/float(6)))
        
        return(anim)
    

    # show 1d logistic regression as classification
    def classification_slider(self,**args):  
        # run logistic regression 
        x = self.x_orig
        y = self.data[:,1]
        
        ##### precomputations #####
        # precompute fits input
        x_fit = np.linspace(np.min(x)-1, np.max(x)+1, 100)
        
        # precompute surface 
        costs1 = [v[0] for v in self.cost_history]
        costs2 = [v[1] for v in self.cost_history]
    
        a = np.linspace(min(costs1),max(costs1))   
        b = np.linspace(min(costs2),max(costs2))    

        s,t = np.meshgrid(a,b)
        shape = np.shape(s)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))

        # generate surface based on given data - done very lazily - recomputed each time
        g = 0
        P = len(y)
        for p in range(0,P):
            g+= np.log(1 + np.exp(-y[p]*(s + t*x[p])))

        # reshape and plot the surface, as well as where the zero-plane is
        s.shape = (shape)
        t.shape = (shape)
        g.shape = (shape)
                
        ##### start plotting #####  
        fig = plt.figure(num=None, figsize=(12,4), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133,projection='3d')

        def show_fit(step):
            ### initialize plot data points and fit
            # initialize fit
            vals = self.cost_history[step]
            b = vals[0]
            w = vals[1]
            
            #### print left panel ####
            # transform input if needed for plotting
            y_fit = np.tanh(b + x_fit*w)

            # plot fit to data
            ax1.cla()
            ax1.plot(x_fit,y_fit,'-k',linewidth = 3,zorder= 0) 

            # initialize points
            pos_inds = np.argwhere(y > 0)
            pos_inds = [v[0] for v in pos_inds]
            ax1.scatter(x[pos_inds],y[pos_inds],color = 'salmon',linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)

            neg_inds = np.argwhere(y < 0)
            neg_inds = [v[0] for v in neg_inds]
            ax1.scatter(x[neg_inds],y[neg_inds],color = 'cornflowerblue',linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)

            # clean up panel
            xgap = float(max(x) - min(x))/float(10)
            ax1.set_xlim([min(x)-xgap,max(x)+xgap])
            ygap = float(max(y) - min(y))/float(10)
            ax1.set_ylim([min(y)-ygap,max(y)+ygap])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('logistic fit')

            #### print middle panel ####
            # plot fit to data
            ax2.cla()
            y_fit = np.sign(b + x_fit*w)
            ax2.plot(x_fit,y_fit,'-k',linewidth = 3,zorder= 0) 

            # initialize points
            pos_inds = np.argwhere(y > 0)
            pos_inds = [v[0] for v in pos_inds]
            ax2.scatter(x[pos_inds],y[pos_inds],color = 'salmon',linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)

            neg_inds = np.argwhere(y < 0)
            neg_inds = [v[0] for v in neg_inds]
            ax2.scatter(x[neg_inds],y[neg_inds],color = 'cornflowerblue',linewidth = 1,marker = 'o',edgecolor = 'k',s = 80)

            # clean up panel
            xgap = float(max(x) - min(x))/float(10)
            ax2.set_xlim([min(x)-xgap,max(x)+xgap])
            ygap = float(max(y) - min(y))/float(10)
            ax2.set_ylim([min(y)-ygap,max(y)+ygap])
            ax2.set_xticks([])
            ax2.set_yticks([])     
            ax2.set_title('final classifier')
            
            #### print right panel ####
            ax3.cla()
            ax3.plot_surface(s,t,g,alpha = 0.15)
            ax3.plot_surface(s,t,g*0,alpha = 0.1)

            # plot all gradient descent steps faintly for visualization purposes
            bs = []
            ws = []
            costs = []
            for i in range(len(self.cost_history)):
                bwg = self.cost_history[i]
                b = bwg[0]
                w = bwg[1]
                cost = bwg[2]
                bs.append(b)
                ws.append(w)
                costs.append(cost)
            ax3.scatter(bs,ws,costs,color = 'm',marker = 'x',linewidth = 3, alpha = 0.1)            

            # plot current gradient descent step in bright red
            b = vals[0]
            w = vals[1]
            cost = vals[2]
            artist = ax3.scatter(b,w,cost,marker = 'o',color = 'r',s = 60,edgecolor = 'k',linewidth = 4)            
            
            # clean up panel
            ax3.view_init(args['view'][0],args['view'][1])        
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_zticks([])
            ax3.set_title('cost function')
            
            ax3.set_xlabel(args['xlabel'],fontsize = 14,labelpad = -5)
            ax3.set_ylabel(args['ylabel'],fontsize = 14,labelpad = -5)

            ax3.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax3.set_zlabel('cost  ',fontsize = 14, rotation = 0,labelpad = 1)
            
            return artist,
        
        anim = animation.FuncAnimation(fig, show_fit,frames=len(self.cost_history), interval=len(self.cost_history), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = int(len(self.cost_history)/float(6)))
        
        return(anim)
        
        