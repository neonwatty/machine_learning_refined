import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
import copy
from inspect import signature

'''
A list of cost functions for supervised learning.  Use the choose_cost function
to choose the desired cost with input data  
'''
class Setup:
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