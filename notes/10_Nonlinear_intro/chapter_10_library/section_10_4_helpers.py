import math, time, copy
from inspect import signature

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

import autograd.numpy as np
from autograd import grad as compute_grad   
from autograd import value_and_grad 
from autograd import hessian as compute_hess
from autograd.misc.flatten import flatten_func

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


    
class ClassificationVisualizer:
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
        fig, axs = plt.subplots(1, 3, figsize=(14,4))

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
        fig = plt.figure(figsize=(14,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        ax1 = plt.subplot(gs[1]);
        
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

        t = np.tanh(model(normalizer(s),w))
        ax1.plot(s.flatten(),t.flatten(),linewidth = 2,c = 'lime')    
         
    # plot regression fits
    def plot_fit_and_feature_space(self,w,model,feat,**kwargs):        
        # construct figure
        fig = plt.figure(figsize=(14,4))

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

        t = np.tanh(model(normalizer(s),w))
        ax1.plot(s.flatten(),t.flatten(),linewidth = 2,c = 'lime')    
        
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
            t = np.tanh(model(normalizer(s),w))
            ax2.plot(s2.flatten(),t.flatten(),linewidth = 2,c = 'lime')    
            
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
            r = np.tanh(np.dot(c.T,w))
            
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
        ax.scatter(x.flatten(),self.y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

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
        
     
class NonlinearVisualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1]

        self.colors = ['cornflowerblue','salmon','lime','bisque','mediumaquamarine','b','m','g']
    
    ######## show N = 1 static image ########
    # show coloring of entire space
    def static_N1_img(self,w_best,cost,predict,**kwargs):
        # or just take last weights
        self.w = w_best
        
        # initialize figure
        fig = plt.figure(figsize = (15,5))
        
        show_cost = False
        if show_cost == True:   
            gs = gridspec.GridSpec(1, 3,width_ratios = [1,1,1],height_ratios = [1]) 
            
            # create third panel for cost values
            ax3 = plt.subplot(gs[2],aspect = 'equal')
            
        else:
            gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 

        #### left plot - data and fit in original space ####
        # setup current axis
        ax = plt.subplot(gs[0]);
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        
        # scatter original points
        self.scatter_pts(ax,self.x)
        ax.set_xlabel(r'$x$', fontsize = 14,labelpad = 10)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
        
        # create fit
        gapx = (max(self.x) - min(self.x))*0.1
        s = np.linspace(min(self.x) - gapx,max(self.x) + gapx,100)
        t = [np.tanh(predict(np.asarray([v]),self.w)) for v in s]
        
        # plot fit
        ax.plot(s,t,c = 'lime')
        ax.axhline(linewidth=0.5, color='k',zorder = 1)

        #### plot data in new space in middle panel (or right panel if cost function decrease plot shown ) #####
        if 'f_x' in kwargs:
            f_x = kwargs['f_x']

            # scatter points
            self.scatter_pts(ax2,f_x)

            # create and plot fit
            s = np.linspace(min(f_x) - 0.1,max(f_x) + 0.1,100)
            t = np.tanh(self.w[0] + self.w[1]*s)
            ax2.plot(s,t,c = 'lime')
            ax2.set_xlabel(r'$f\,(x)$', fontsize = 14,labelpad = 10)
            ax2.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 10)
        
        if 'f2_x' in kwargs:
            ax2 = plt.subplot(gs[1],projection = '3d');   
            view = kwargs['view']
            
            # get input
            f1_x = kwargs['f1_x']
            f2_x = kwargs['f2_x']

            # scatter points
            f1_x = np.asarray(f1_x)
            f1_x.shape = (len(f1_x),1)
            f2_x = np.asarray(f2_x)
            f2_x.shape = (len(f2_x),1)
            xtran = np.concatenate((f1_x,f2_x),axis = 1)
            self.scatter_pts(ax2,xtran)

            # create and plot fit
            s1 = np.linspace(min(f1_x) - 0.1,max(f1_x) + 0.1,100)
            s2 = np.linspace(min(f2_x) - 0.1,max(f2_x) + 0.1,100)
            t1,t2 = np.meshgrid(s1,s2)
            
            # compute fitting hyperplane
            t1.shape = (len(s1)**2,1)
            t2.shape = (len(s2)**2,1)
            r = np.tanh(self.w[0] + self.w[1]*t1 + self.w[2]*t2)
            
            # reshape for plotting
            t1.shape = (len(s1),len(s1))
            t2.shape = (len(s2),len(s2))
            r.shape = (len(s1),len(s2))
            ax2.plot_surface(t1,t2,r,alpha = 0.1,color = 'lime',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')
                
            # label axes
            self.move_axis_left(ax2)
            ax2.set_xlabel(r'$f_1(x)$', fontsize = 12,labelpad = 5)
            ax2.set_ylabel(r'$f_2(x)$', rotation = 0,fontsize = 12,labelpad = 5)
            ax2.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
            ax2.view_init(view[0],view[1])
            
        # plot cost function decrease
        if  show_cost == True: 
            # compute cost eval history
            g = cost
            cost_evals = []
            for i in range(len(w_hist)):
                W = w_hist[i]
                cost = g(W)
                cost_evals.append(cost)
     
            # plot cost path - scale to fit inside same aspect as classification plots
            num_iterations = len(w_hist)
            minx = min(self.x)
            maxx = max(self.x)
            gapx = (maxx - minx)*0.1
            minc = min(cost_evals)
            maxc = max(cost_evals)
            gapc = (maxc - minc)*0.1
            minc -= gapc
            maxc += gapc
            
            s = np.linspace(minx + gapx,maxx - gapx,num_iterations)
            scaled_costs = [c/float(max(cost_evals))*(maxx-gapx) - (minx+gapx) for c in cost_evals]
            ax3.plot(s,scaled_costs,color = 'k',linewidth = 1.5)
            ax3.set_xlabel('iteration',fontsize = 12)
            ax3.set_title('cost function plot',fontsize = 12)
            
            # rescale number of iterations and cost function value to fit same aspect ratio as other two subplots
            ax3.set_xlim(minx,maxx)
            #ax3.set_ylim(minc,maxc)
            
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

    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_img(self,w_best,runner,**kwargs):
        cost = runner.cost_func
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
                
        # count parameter layers of input to feature transform
        sig = signature(feat)
        sig = len(sig.parameters)

        # or just take last weights
        self.w = w_best
        
        zplane = 'on'
        if 'zplane' in kwargs:
            zplane = kwargs['zplane']
        view1 = [20,45]
        if 'view1' in kwargs:
            view1 = kwargs['view1']
        view2 = [20,30]
        if 'view2' in kwargs:
            view2 = kwargs['view2']  
            
        # initialize figure
        fig = plt.figure(figsize = (10,9))
        gs = gridspec.GridSpec(2, 2,width_ratios = [1,1]) 

        #### left plot - data and fit in original space ####
        # setup current axis
        ax = plt.subplot(gs[0],aspect = 'equal');
        ax2 = plt.subplot(gs[1],aspect = 'equal');
        ax3 = plt.subplot(gs[2],projection = '3d');
        ax4 = plt.subplot(gs[3],projection = '3d');
        
        ### cleanup left plots, create max view ranges ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
        ax.set_xlim([xmin1,xmax1])
        ax3.set_xlim([xmin1,xmax1])

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
        ax.set_ylim([xmin2,xmax2])
        ax3.set_ylim([xmin2,xmax2])

        ymin = np.min(self.y)
        ymax = np.max(self.y)
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap
        ax3.set_zlim([ymin,ymax])
        
        ax3.axis('off')
        ax3.view_init(view1[0],view1[1])

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            
        #### plot left panels ####
        # plot points in 2d and 3d
        ind0 = np.argwhere(self.y == +1)
        ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')
        ax3.scatter(self.x[ind0,0],self.x[ind0,1],self.y[ind0],s = 55, color = self.colors[0], edgecolor = 'k')

        ind1 = np.argwhere(self.y == -1)
        ax.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
        ax3.scatter(self.x[ind1,0],self.x[ind1,1],self.y[ind1],s = 55, color = self.colors[1], edgecolor = 'k')
       
        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,100)
        r2 = np.linspace(xmin2,xmax2,100)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        z = predict(normalizer(h.T),self.w)
        z = np.tanh(z)
        
        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
        ax3.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')

        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            # plot zplane
            ax3.plot_surface(s,t,z*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'w',edgecolor = 'k') 
            
            # plot separator curve in left plot
            ax3.contour(s,t,z,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            ax3.contourf(s,t,z,colors = self.colors[1],levels = [0,1],zorder = 1,alpha = 0.1)
            ax3.contourf(s,t,z+1,colors = self.colors[0],levels = [0,1],zorder = 1,alpha = 0.1)
        
        #### plot right panel scatter ####
        # transform data
        f = 0 
        if sig == 1:
            f = feat(normalizer(self.x.T)).T
        else:
            f = feat(normalizer(self.x.T),self.w[0]).T
        x1 = f[:,0]
        x2 = f[:,1]
        #x1 = [f1(e) for e in self.x]
        #x2 = [f2(e) for e in self.x]
        ind0 = [v[0] for v in ind0]
        ind1 = [v[0] for v in ind1]

        # plot points on desired panel
        v1 = [x1[e] for e in ind0]
        v2 = [x2[e] for e in ind0]
        ax2.scatter(v1,v2,s = 55, color = self.colors[0], edgecolor = 'k')
        ax4.scatter(v1,v2,self.y[ind0],s = 55, color = self.colors[0], edgecolor = 'k')

        v1 = [x1[e] for e in ind1]
        v2 = [x2[e] for e in ind1]        
        ax2.scatter(v1,v2,s = 55, color = self.colors[1], edgecolor = 'k')
        ax4.scatter(v1,v2,self.y[ind1],s = 55, color = self.colors[1], edgecolor = 'k')
        
        ### cleanup right panels - making max viewing ranges ###
        
        xmin1 = np.min(x1)
        xmax1 = np.max(x1)
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1
        ax2.set_xlim([xmin1,xmax1])
        ax4.set_xlim([xmin1,xmax1])

        xmin2 = np.min(x2)
        xmax2 = np.max(x2)
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2
        
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(r'$f\,_1\left(\mathbf{x}\right)$',fontsize = 15)
        ax2.set_ylabel(r'$f\,_2\left(\mathbf{x}\right)$',fontsize = 15)        
        
        ### plot right panel 3d scatter ###
        #### make right plot contour ####
        r1 = np.linspace(xmin1,xmax1,100)
        r2 = np.linspace(xmin2,xmax2,100)
        s,t = np.meshgrid(r1,r2)
        
        s.shape = (1,len(r1)**2)
        t.shape = (1,len(r2)**2)
       # h = np.vstack((s,t))
       # h = feat(normalizer(h))  
       # s = h[0,:]
       # t = h[1,:]
        z = 0
        if sig == 1:
            z = self.w[0] + self.w[1]*s + self.w[2]*t
        else:
            z = self.w[1][0] + self.w[1][1]*s + self.w[1][2]*t
        z = np.tanh(np.asarray(z))
        
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))
        z.shape = (np.size(r1),np.size(r2))

        ax2.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax2.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

        #### plot right surface plot ####
        # plot regression surface
        ax4.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=10, cstride=10,linewidth=0.5,edgecolor = 'k')
            
        # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
        if zplane == 'on':
            # plot zplane
            ax4.plot_surface(s,t,z*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'w',edgecolor = 'k') 
            
            # plot separator curve in left plot
            ax4.contour(s,t,z,colors = 'k',levels = [0],linewidths = 3,zorder = 1)
            ax4.contourf(s,t,z,colors = self.colors[0],levels = [0,1],zorder = 1,alpha = 0.1)
            ax4.contourf(s,t,z+1,colors = self.colors[1],levels = [0,1],zorder = 1,alpha = 0.1)
   
        ax2.set_ylim([xmin2,xmax2])
        ax4.set_ylim([xmin2,xmax2])

        ax4.axis('off')
        ax4.view_init(view2[0],view2[1])
        ax4.set_zlim([ymin,ymax])

    ######## show N = 2 static image ########
    # show coloring of entire space
    def static_N2_simple(self,w_best,runner,**kwargs):
        cost = runner.cost_func
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
                
        # count parameter layers of input to feature transform
        sig = signature(feat)
        sig = len(sig.parameters)

        # or just take last weights
        self.w = w_best

        # construct figure
        fig, axs = plt.subplots(1, 2, figsize=(14,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2) 
        ax2 = plt.subplot(gs[1],aspect = 'equal'); 
        ax1 = plt.subplot(gs[0],projection='3d'); 

        # scatter points
        self.scatter_pts(ax1,self.x)

        ### from above
        ax2.set_xlabel(r'$x_1$',fontsize = 15)
        ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # plot points in 2d and 3d
        C = len(np.unique(self.y))
        if C == 2:
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
        else:
            for c in range(C):
                ind0 = np.argwhere(self.y == c)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)

        ### create surface and boundary plot ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    
        if 'view' in kwargs:
            view = kwargs['view']
            ax1.view_init(view[0],view[1])

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,300)
        r2 = np.linspace(xmin2,xmax2,300)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        z = predict(normalizer(h.T),self.w)
        z = np.sign(z)
        
        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax2.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax2.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))
        ax1.plot_surface(s,t,z,alpha = 0.1,color = 'w',rstride=30, cstride=30,linewidth=0.5,edgecolor = 'k')

            
    ###### plot plotting functions ######
    def plot_data(self):
        fig = 0
        # plot data in two and one-d
        if np.shape(self.x)[1] < 2:
            # construct figure
            fig, axs = plt.subplots(2,1, figsize=(4,4))
            gs = gridspec.GridSpec(2,1,height_ratios = [6,1]) 
            ax1 = plt.subplot(gs[0],aspect = 'equal');
            ax2 = plt.subplot(gs[1],sharex = ax1); 
            
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.05
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.05
            ymin -= ygap
            ymax += ygap    

            ### plot in 2-d
            ax1.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.axhline(linewidth=0.5, color='k',zorder = 1)
            
            ### plot in 1-d
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0],np.zeros((len(self.x[ind0]))),s = 55, color = self.colors[1], edgecolor = 'k',zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1],np.zeros((len(self.x[ind1]))),s = 55, color = self.colors[0], edgecolor = 'k',zorder = 3)
            ax2.set_yticks([0])
            ax2.axhline(linewidth=0.5, color='k',zorder = 1)
        
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            
        if np.shape(self.x)[1] == 2:
            
            # initialize figure
            fig = plt.figure(figsize = (9,3))
            gs = gridspec.GridSpec(1, 2,width_ratios = [1,1]) 
            ax2 = plt.subplot(gs[1],aspect = 'equal');
            ax1 = plt.subplot(gs[0],projection = '3d');

            # scatter points
            self.scatter_pts(ax1,self.x)
            
            ### from above
            ax2.set_xlabel(r'$x_1$',fontsize = 15)
            ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # plot points in 2d and 3d
            C = len(np.unique(self.y))
            if C == 2:
                ind0 = np.argwhere(self.y == +1)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[1], edgecolor = 'k')

                ind1 = np.argwhere(self.y == -1)
                ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[0], edgecolor = 'k')
            else:
                for c in range(C):
                    ind0 = np.argwhere(self.y == c)
                    ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 50, color = self.colors[c], edgecolor = 'k',linewidth=2)
                    
                  
        
            self.move_axis_left(ax1)
            ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
            view = [20,45]
            ax1.view_init(view[0],view[1])
            
    # scatter points
    def scatter_pts(self,ax,x):
        if np.shape(x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(x))
            xmin = copy.deepcopy(min(x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
        if np.shape(x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(x[:,0]))
            xmin1 = copy.deepcopy(min(x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.1
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(x[:,1]))
            xmin2 = copy.deepcopy(min(x[:,1]))
            xgap2 = (xmax2 - xmin2)*0.1
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x[:,0],x[:,1],self.y.flatten(),s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1), round(xmax1)+1, 1.0))
            ax.set_yticks(np.arange(round(xmin2), round(xmax2)+1, 1.0))
            ax.set_zticks(np.arange(round(ymin), round(ymax)+1, 1.0))
           
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
        
        
    # toy plot
    def multiclass_plot(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        
        # grab args
        view = [20,-70]
        if 'view' in kwargs:
            view = kwargs['view']
 
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,600)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1).T

        g_vals = model(normalizer(h),w)
        g_vals = np.asarray(g_vals)
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # create figure to plot
        fig = plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')

        ### create 3d plot in left panel
        #ax1 = plt.subplot(121,projection = '3d')
        ax2 = plt.subplot(132)

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        # scatter points in both panels
        class_nums = np.unique(self.y)
        C = len(class_nums)
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            #ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 80,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 110,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
        # switch for 2class / multiclass view
        if C == 2:
            # plot regression surface
            #ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'k',rstride=20, cstride=20,linewidth=0,edgecolor = 'k') 

            # plot zplane = 0 in left 3d panel - showing intersection of regressor with z = 0 (i.e., its contour, the separator, in the 3d plot too)?
            #ax1.plot_surface(w1_vals,w2_vals,g_vals*0,alpha = 0.1,rstride=20, cstride=20,linewidth=0.15,color = 'k',edgecolor = 'k') 
            
            # plot separator in left plot z plane
            #ax1.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            #ax1.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,2))
            #ax1.contourf(w1_vals,w2_vals,-g_vals+1,colors = self.colors[1:],alpha = 0.1,levels = range(0,2))
    
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+1,colors = self.colors[:],alpha = 0.1,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot           
            minz = min(copy.deepcopy(self.y))
            maxz = max(copy.deepcopy(self.y))

            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz

        
        # multiclass view
        else:   
            '''
            ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

            for c in range(C):
                # plot separator curve in left plot z plane
                ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

                # color parts of plane with correct colors
                ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
            '''
            
            # plot separator in right plot
            ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)
            
            # adjust height of regressor to plot filled contours
            ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

            ### clean up panels
            # set viewing limits on vertical dimension for 3d plot 
            minz = 0
            maxz = max(copy.deepcopy(self.y))
            gapz = (maxz - minz)*0.1
            minz -= gapz
            maxz += gapz
            #ax1.set_zlim([minz,maxz])

            #ax1.view_init(view[0],view[1]) 

        '''
        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)
        '''
        
        ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
        ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        
    # toy plot
    def show_individual_classifiers(self,run,w,**kwargs):
        model = run.model
        normalizer = run.normalizer
        feat = run.feature_transforms
        
        # grab args
        view = [20,-70]
        if 'view' in kwargs:
            view = kwargs['view']
 
        ### plot all input data ###
        # generate input range for functions
        minx = min(min(self.x[:,0]),min(self.x[:,1]))
        maxx = max(max(self.x[:,0]),max(self.x[:,1]))
        gapx = (maxx - minx)*0.1
        minx -= gapx
        maxx += gapx

        r = np.linspace(minx,maxx,600)
        w1_vals,w2_vals = np.meshgrid(r,r)
        w1_vals.shape = (len(r)**2,1)
        w2_vals.shape = (len(r)**2,1)
        h = np.concatenate([w1_vals,w2_vals],axis = 1).T

        g_vals = model(normalizer(h),w)
        g_vals = np.asarray(g_vals)
        g_new = copy.deepcopy(g_vals).T
        g_vals = np.argmax(g_vals,axis = 0)

        # vals for cost surface
        w1_vals.shape = (len(r),len(r))
        w2_vals.shape = (len(r),len(r))
        g_vals.shape = (len(r),len(r))

        # count points
        class_nums = np.unique(self.y)
        C = int(len(class_nums))
        
        fig = plt.figure(figsize = (10,7))
        gs = gridspec.GridSpec(2, C) 

        #### left plot - data and fit in original space ####
        # setup current axis
        #ax1 = plt.subplot(gs[C],projection = '3d');
        ax2 = plt.subplot(gs[C+1],aspect = 'equal');
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure

        ##### plot top panels ####
        for d in range(C):
            # create panel
            ax = plt.subplot(gs[d],aspect = 'equal'); ax.axis('off')
                       
            for c in range(C):
                # plot points
                ind = np.argwhere(self.y == class_nums[c])
                ind = [v[0] for v in ind]
                ax.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
            
            g_2 = np.sign(g_new[:,d])
            g_2.shape = (len(r),len(r))

            # plot separator curve 
            ax.contour(w1_vals,w2_vals,g_2+1,colors = 'k',levels = [-1,1],linewidths = 4.5,zorder = 1,linestyle = '-')
            ax.contour(w1_vals,w2_vals,g_2+1,colors = self.colors[d],levels = [-1,1],linewidths = 2.5,zorder = 1,linestyle = '-')
                
            ax.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
            ax.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)
        
        ##### plot bottom panels ###
        # scatter points in both bottom panels
        for c in range(C):
            ind = np.argwhere(self.y == class_nums[c])
            ind = [v[0] for v in ind]
            #ax1.scatter(self.x[ind,0],self.x[ind,1],self.y[ind],s = 50,color = self.colors[c],edgecolor = 'k',linewidth = 1.5)
            ax2.scatter(self.x[ind,0],self.x[ind,1],s = 50,color = self.colors[c],edgecolor = 'k', linewidth = 2)
      
        #ax1.plot_surface(w1_vals,w2_vals,g_vals,alpha = 0.1,color = 'w',rstride=45, cstride=45,linewidth=0.25,edgecolor = 'k')

        #for c in range(C):
            # plot separator curve in left plot z plane
            #ax1.contour(w1_vals,w2_vals,g_vals - c,colors = 'k',levels = [0],linewidths = 3,zorder = 1)

            # color parts of plane with correct colors
            #ax1.contourf(w1_vals,w2_vals,g_vals - c +0.5,colors = self.colors[c],alpha = 0.4,levels = [0,1])
             
        # plot separator in right plot
        ax2.contour(w1_vals,w2_vals,g_vals,colors = 'k',levels = range(0,C+1),linewidths = 3,zorder = 1)

        # adjust height of regressor to plot filled contours
        ax2.contourf(w1_vals,w2_vals,g_vals+0.5,colors = self.colors[:],alpha = 0.2,levels = range(0,C+1))

        ### clean up panels
        # set viewing limits on vertical dimension for 3d plot 
        minz = 0
        maxz = max(copy.deepcopy(self.y))
        gapz = (maxz - minz)*0.1
        minz -= gapz
        maxz += gapz
        ax2.axis('off')
        
        '''
        ax1.set_zlim([minz,maxz])

        ax1.view_init(view[0],view[1]) 

        # clean up panel
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        ax1.xaxis.pane.set_edgecolor('white')
        ax1.yaxis.pane.set_edgecolor('white')
        ax1.zaxis.pane.set_edgecolor('white')

        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        self.move_axis_left(ax1)
        ax1.set_xlabel(r'$x_1$', fontsize = 16,labelpad = 5)
        ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 16,labelpad = 5)
        ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 5)
        '''
        
       # ax2.set_xlabel(r'$x_1$', fontsize = 18,labelpad = 10)
       # ax2.set_ylabel(r'$x_2$', rotation = 0,fontsize = 18,labelpad = 15)