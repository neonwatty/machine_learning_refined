import math, time, copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib import gridspec
from inspect import signature
from matplotlib.ticker import FormatStrFormatter

import autograd.numpy as np
from autograd import grad as compute_grad   
from autograd import value_and_grad 
from autograd import hessian as compute_hess
from autograd.misc.flatten import flatten_func


class StaticVisualizer:
    '''
    Illustrate a run of your preferred optimization algorithm on a one or two-input function.  Run
    the algorithm first, and input the resulting weight history into this wrapper.
    ''' 

    # compare cost histories from multiple runs
    def plot_cost_histories(self,histories,start,**kwargs):
        # plotting colors
        colors = ['k','magenta','aqua','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']
            
        # plot points on cost function plot too?
        points = False
        if 'points' in kwargs:
            points = kwargs['points']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(histories)):
            history = histories[c]
            label = 0
            if c == 0:
                label = labels[0]
            else:
                label = labels[1]
                
            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c]) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 
                
            # check if points should be plotted for visualization purposes
            if points == True:
                ax.scatter(np.arange(start,len(history),1),history[start:],s = 90,color = colors[c],edgecolor = 'w',linewidth = 2,zorder = 3) 


        # clean up panel
        xlabel = 'step $k$'
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        if np.size(label) > 0:
            anchor = (1,1)
            if 'anchor' in kwargs:
                anchor = kwargs['anchor']
            plt.legend(loc='upper right', bbox_to_anchor=anchor)
            #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xlim([start - 0.5,len(history) - 0.5])
        
       # fig.tight_layout()
        plt.show()

       

'''
A list of cost functions for supervised learning.  Use the choose_cost function
to choose the desired cost with input data  
'''
class BaseSetup:
    def __init__(self,x,y,feature_transforms,cost,**kwargs):
        normalize = 'standard'
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        if normalize == 'standard':
            # create normalizer
            self.normalizer,self.inverse_normalizer = self.standard_normalizer(x)

            # normalize input 
            self.x = self.normalizer(x)
        elif normalize == 'sphere':
            # create normalizer
            self.normalizer,self.inverse_normalizer = self.PCA_sphereing(x)

            # normalize input 
            self.x = self.normalizer(x)
        else:
            self.x = x
            self.normalizer = lambda data: data
            self.inverse_normalizer = lambda data: data
            
        # make any other variables not explicitly input into cost functions globally known
        self.y = y
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)

        self.lam = 0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']

        # make cost function choice
        cost_func = 0
        if cost == 'least_squares':
            self.cost_func = self.least_squares
        if cost == 'least_absolute_deviations':
            self.cost_func = self.least_absolute_deviations
        if cost == 'softmax':
            self.cost_func = self.softmax
        if cost == 'relu':
            self.cost_func = self.relu
        if cost == 'counter':
            self.cost_func = self.counting_cost
        if cost == 'multiclass_perceptron':
            self.cost_func = self.multiclass_perceptron
        if cost == 'multiclass_softmax':
            self.cost_func = self.multiclass_softmax
        if cost == 'multiclass_counter':
            self.cost_func = self.multiclass_counting_cost
            
        # for autoencoder
        if cost == 'autoencoder':
            self.feature_transforms_2 = kwargs['feature_transforms_2']
            self.cost_func = self.autoencoder

    # run optimization
    def fit(self,**kwargs):
        # basic parameters for gradient descent run
        max_its = 500; alpha_choice = 10**(-1);
        w = 0.1*np.random.randn(np.shape(self.x)[0] + 1,1)
        algo = 'gradient_descent'

        # set parameters by hand
        if 'algo' in kwargs:
            algo = kwargs['algo']
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            alpha_choice = kwargs['alpha_choice']
        if 'w' in kwargs:
            w = kwargs['w']

        # run gradient descent
        if algo == 'gradient_descent':
            self.weight_history, self.cost_history = self.gradient_descent(self.cost_func,alpha_choice,max_its,w)
        if algo == 'newtons_method':  
            self.weight_history, self.cost_history = self.newtons_method(self.cost_func,max_its,w)

    ###### cost functions #####
    # compute linear combination of input point
    def model(self,x,w):   
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        f = 0
        if len(self.sig.parameters) == 2:
            f = self.feature_transforms(x,w[0])
        else: 
            f = self.feature_transforms(x)    

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if len(self.sig.parameters) == 2:
            a = w[1][0] + np.dot(f.T,w[1][1:])
        else:
            a = w[0] + np.dot(f.T,w[1:])
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(np.size(self.y))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w):
        cost = np.sum(np.abs(self.model(self.x,w) - self.y))
        return cost/float(np.size(self.y))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(np.size(self.y))

    # the convex relu cost function
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*self.model(self.x,w)))
        return cost/float(np.size(self.y))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute maximum across data points
        a = np.max(all_evals,axis = 0)    

        # compute cost in compact form using numpy broadcasting
        b = all_evals[self.y.astype(int).flatten(),np.arange(np.size(self.y))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(self.y))

    # multiclass softmax
    def multiclass_softmax(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[self.y.astype(int).flatten(),np.arange(np.size(self.y))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(self.y))
    
    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w):                
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        # return number of misclassifications
        return count
    
    ### for autoencoder ###
    def encoder(self,x,w):    
        # feature transformation 
        f = self.feature_transforms(x,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T

    def decoder(self,v,w):
        # feature transformation 
        f = self.feature_transforms_2(v,w[0])

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        a = np.dot(f.T,w[1])
        return a.T
    
    def autoencoder(self,w):
        # encode input
        a = self.encoder(self.x,w[0])
        
        # decode result
        b = self.decoder(a,w[1])
        
        # compute Least Squares error
        cost = np.sum((b - self.x)**2)
        return cost/float(self.x.shape[1])
    
    ##### optimizer ####
    # gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
    def gradient_descent(self,g,alpha_choice,max_its,w):
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
 
    # newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
    def newtons_method(self,g,max_its,w,**kwargs):
        # flatten input funciton, in case it takes in matrices of weights
        flat_g, unflatten, w = flatten_func(g, w)

        # compute the gradient / hessian functions of our input function -
        # note these are themselves functions.  In particular the gradient - 
        # - when evaluated - returns both the gradient and function evaluations (remember
        # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
        # an Automatic Differntiator to evaluate the gradient)
        gradient = value_and_grad(flat_g)
        hess = hessian(flat_g)

        # set numericxal stability parameter / regularization parameter
        epsilon = 10**(-7)
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']

        # run the newtons method loop
        weight_history = []      # container for weight history
        cost_history = []        # container for corresponding cost function history
        for k in range(max_its):
            # evaluate the gradient, store current weights and cost function value
            cost_eval,grad_eval = gradient(w)
            weight_history.append(unflatten(w))
            cost_history.append(cost_eval)

            # evaluate the hessian
            hess_eval = hess(w)

            # reshape for numpy linalg functionality
            hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

            # solve second order system system for weight update
            w = w - np.dot(np.linalg.pinv(hess_eval + epsilon*np.eye(np.size(w))),grad_eval)

        # collect final weights
        weight_history.append(unflatten(w))
        # compute final cost function value via g itself (since we aren't computing 
        # the gradient at the final step we don't get the final cost function value 
        # via the Automatic Differentiatoor) 
        cost_history.append(flat_g(w))  

        return weight_history,cost_history

    ###### normalizers #####
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    # compute eigendecomposition of data covariance matrix
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        D,V = np.linalg.eigh(Cov)
        return D,V

    # PCA-sphereing - use PCA to normalize input features
    def PCA_sphereing(self,x,**kwargs):
        # Step 1: mean-center the data
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_centered = x - x_means

        # Step 2: compute pca transform on mean-centered data
        d,V = self.PCA(x_centered,**kwargs)

        # Step 3: divide off standard deviation of each (transformed) input, 
        # which are equal to the returned eigenvalues in 'd'.  
        stds = (d[:,np.newaxis])**(0.5)
        normalizer = lambda data: np.dot(V.T,data - x_means)/stds

        # create inverse normalizer
        inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

        # return normalizer 
        return normalizer,inverse_normalizer



class RegressionVisualizer:
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
    def plot_fit(self,w,model,**kwargs):        
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off') 
        ax = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
  
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
 
        normalizer = lambda a: a
        if 'normalizer' in kwargs:
            normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        ax.plot(s.T,t.T,linewidth = 4,c = 'k',zorder = 0)
        ax.plot(s.T,t.T,linewidth = 2,c = 'lime',zorder = 0)
         
    # plot regression fits
    def plot_fit_and_feature_space(self,w,model,feat,**kwargs):        
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1]); 
        
        view = [20,20]
        if 'view' in kwargs:
            view = kwargs['view']

        ##### plot left panel in original space ####
        # scatter points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax1)

        # clean up panel
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])

        # label axes
        ax1.set_xlabel(r'$x$', fontsize = 16)
        ax1.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 10)
        
        # create fit
        s = np.linspace(xmin,xmax,300)[np.newaxis,:]
 
        normalizer = lambda a: a
        if 'normalizer' in kwargs:
            normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        
        ax1.plot(s.flatten(),t.flatten(),linewidth = 4,c = 'k',zorder = 0)    
        ax1.plot(s.flatten(),t.flatten(),linewidth = 2,c = 'lime',zorder = 0)

        #### plot fit in transformed feature space #####
        # check if feature transform has internal parameters
        x_transformed = 0
        sig = signature(feat)
        if len(sig.parameters) == 2:
            if np.shape(w)[1] == 1:
                x_transformed = feat(normalizer(self.x),w)
            else:
                x_transformed = feat(normalizer(self.x),w[0])
        else: 
            x_transformed = feat(normalizer(self.x))
        
        # two dimensional transformed feature space
        if x_transformed.shape[0] == 1:
            s = np.linspace(xmin,xmax,300)[np.newaxis,:]
            
            # scatter points
            xmin,xmax,ymin,ymax = self.scatter_pts_2d(x_transformed,ax2)
        
            # produce plot
            s2 = copy.deepcopy(s)
            if len(sig.parameters) == 2:
                if np.shape(w)[1] == 1:
                    s2 = feat(normalizer(s),w)
                else:
                    s2 = feat(normalizer(s),w[0])
            else: 
                s2 = feat(normalizer(s))
            t = model(normalizer(s),w)
            
            ax2.plot(s2.flatten(),t.flatten(),linewidth = 4,c = 'k',zorder = 0)    
            ax2.plot(s2.flatten(),t.flatten(),linewidth = 2,c = 'lime',zorder = 0)
            
            # label axes
            ax2.set_xlabel(r'$f\left(x,\mathbf{w}^{\star}\right)$', fontsize = 16)
            ax2.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 10)
            
        # three dimensional transformed feature space
        if x_transformed.shape[0] == 2:
            # create panel
            ax2 = plt.subplot(gs[1],projection = '3d');  
            s = np.linspace(xmin,xmax,100)[np.newaxis,:]

            # plot data in 3d
            xmin,xmax,xmin1,xmax1,ymin,ymax = self.scatter_3d_points(x_transformed,ax2)

            # create and plot fit
            s2 = copy.deepcopy(s)
            if len(sig.parameters) == 2:
                s2 = feat(normalizer(s),w[0])
            else: 
                s2 = feat(normalizer(s))
 
            # reshape for plotting
            a = s2[0,:]
            b = s2[1,:]
            a = np.linspace(xmin,xmax,100)
            b = np.linspace(xmin1,xmax1,100)
            a,b = np.meshgrid(a,b)
            
            # get firstem
            a.shape = (1,np.size(s)**2)
            f1 = feat(normalizer(a))[0,:]
            
            # secondm
            b.shape = (1,np.size(s)**2)
            f2 = feat(normalizer(b))[1,:]
            
            # tack a 1 onto the top of each input point all at once
            c = np.vstack((a,b))
            o = np.ones((1,np.shape(c)[1]))
            c = np.vstack((o,c))
            r = (np.dot(c.T,w))
            
            # various
            a.shape = (np.size(s),np.size(s))
            b.shape = (np.size(s),np.size(s))
            r.shape = (np.size(s),np.size(s))
            ax2.plot_surface(a,b,r,alpha = 0.1,color = 'lime',rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k')
            ax2.set_xlim([np.min(a),np.max(a)])
            ax2.set_ylim([np.min(b),np.max(b)])
            
            '''
            a,b = np.meshgrid(t1,t2)
            a.shape = (1,np.size(s)**2)
            b.shape = (1,np.size(s)**2)
            '''
 
            '''
            c = np.vstack((a,b))
            o = np.ones((1,np.shape(c)[1]))
            c = np.vstack((o,c))

            # tack a 1 onto the top of each input point all at once
            r = (np.dot(c.T,w))

            a.shape = (np.size(s),np.size(s))
            b.shape = (np.size(s),np.size(s))
            r.shape = (np.size(s),np.size(s))
            ax2.plot_surface(a,b,r,alpha = 0.1,color = 'lime',rstride=15, cstride=15,linewidth=0.5,edgecolor = 'k')
            '''
            
            # label axes
            #self.move_axis_left(ax2)
            ax2.set_xlabel(r'$f_1(x)$', fontsize = 12,labelpad = 5)
            ax2.set_ylabel(r'$f_2(x)$', rotation = 0,fontsize = 12,labelpad = 5)
            ax2.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = 0)
            self.move_axis_left(ax2)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.view_init(view[0],view[1])
 
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
        ax.scatter(x.flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 60,zorder = 3)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        return xmin,xmax,ymin,ymax
    
    ### visualize the surface plot of cost function ###
    def scatter_3d_points(self,x,ax):
        # set plotting limits
        xmax = copy.deepcopy(np.max(x[0,:]))
        xmin = copy.deepcopy(np.min(x[0,:]))
        xgap = (xmax - xmin)*0.2
        xmin -= xgap
        xmax += xgap
        
        xmax1 = copy.deepcopy(np.max(x[1,:]))
        xmin1 = copy.deepcopy(np.min(x[1,:]))
        xgap1 = (xmax1 - xmin1)*0.2
        xmin1 -= xgap1
        xmax1 += xgap1

        ymax = copy.deepcopy(np.max(self.y))
        ymin = copy.deepcopy(np.min(self.y))
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap   
        
        # plot data
        ax.scatter(x[0,:].flatten(),x[1,:].flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)
            
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
        
        
        return xmin,xmax,xmin1,xmax1,ymin,ymax

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