import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
import matplotlib.pyplot as plt
from matplotlib import gridspec
from inspect import signature

class HistoryPlotter:
    def __init__(self,cost_histories,count_histories,start,labels):
        # just plot cost history?
        if len(count_histories) == 0:
            self.plot_cost_histories(cost_histories,start,labels)
        else: # plot cost and count histories
            self.plot_cost_count_histories(cost_histories,count_histories,start,labels)
 
    #### compare cost function histories ####
    def plot_cost_histories(self,cost_histories,start,labels):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(cost_histories)):
            history = cost_histories[c]
            label = labels[c]
                
            # plot cost function history
            ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax.set_title(title,fontsize = 18)
        
        # plot legend
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax.set_xlim([start - 0.5,len(history) - 0.5]) 
        plt.show()
        
    #### compare multiple histories of cost and misclassification counts ####
    def plot_cost_count_histories(self,cost_histories,count_histories,start,labels):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 2) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(cost_histories)):
            cost_history = cost_histories[c]
            count_history = count_histories[c]
            label = labels[c]

            # check if a label exists, if so add it to the plot
            ax1.plot(np.arange(start,len(cost_history),1),cost_history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 
            
            ax2.plot(np.arange(start,len(count_history),1),count_history[start:],linewidth = 3*(0.8)**(c),color = colors[c],label = label) 

        # clean up panel
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax1.set_xlabel(xlabel,fontsize = 14)
        ax1.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax1.set_title(title,fontsize = 18)

        ylabel = 'num misclasses'
        ax2.set_xlabel(xlabel,fontsize = 14)
        ax2.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 10)
        title = 'misclassification history'
        ax2.set_title(title,fontsize = 18)
        
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax1.set_xlim([start - 0.5,len(cost_history) - 0.5])
        ax2.set_xlim([start - 0.5,len(cost_history) - 0.5])
        plt.show()    
        
        
class CostFunction:
    def __init__(self,name,x,y,feature_transforms,**kwargs):
        # point to input/output for cost functions
        self.x = x
        self.y = y
       
        # make copy of feature transformation
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)
        
        ### make cost function choice ###
        # for regression
        if name == 'least_squares':
            self.cost = self.least_squares
        if name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # for two-class classification
        if name == 'softmax':
            self.cost = self.softmax
        if name == 'perceptron':
            self.cost = self.perceptron
        if name == 'twoclass_counter':
            self.cost = self.counting_cost
            
        # for multiclass classification
        if name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if name == 'multiclass_counter':
            self.cost = self.multiclass_counting_cost
            
        # for autoencoder
        if name == 'autoencoder':
            self.feature_transforms = feature_transforms
            self.feature_transforms_2 = kwargs['feature_transforms_2']
            self.cost = self.autoencoder

            
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

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if len(self.sig.parameters) == 2:
            a = np.dot(f.T,w[1])
        else:
            a = np.dot(f.T,w)
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum((self.model(x_p,w) - y_p)**2)
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.abs(self.model(x_p,w) - y_p))
        return cost/float(np.size(y_p))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*self.model(x_p,w))))
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,iter):
        # get batch of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        return cost/float(np.size(y_p))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,iter):
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 0)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

    # multiclass softmax
    def multiclass_softmax(self,w,iter):     
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[:,iter]
        
        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

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
        
        
# minibatch gradient descent
def gradient_descent(g, alpha, max_its, w, num_pts, batch_size,**kwargs):    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
    cost_hist = [g_flat(w,np.arange(num_pts))]
   
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_pts, batch_size)))
    # over the line
    for k in range(max_its):   
        # loop over each minibatch
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_pts))

            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # take descent step with momentum
            w = w - alpha*grad_eval

        # record weight update
        w_hist.append(unflatten(w))
        cost_hist.append(g_flat(w,np.arange(num_pts)))
    return w_hist,cost_hist


class MultilayerPerceptron:
    def __init__(self,**kwargs):        
        # set default values for layer sizes, activation, and scale
        activation = 'relu'

        # decide on these parameters via user input
        if 'activation' in kwargs:
            activation = kwargs['activation']

        # switches
        if activation == 'linear':
            self.activation = lambda data: data
        elif activation == 'tanh':
            self.activation = lambda data: np.tanh(data)
        elif activation == 'relu':
            self.activation = lambda data: np.maximum(0,data)
        elif activation == 'sinc':
            self.activation = lambda data: np.sinc(data)
        elif activation == 'sin':
            self.activation = lambda data: np.sin(data)
        else: # user-defined activation
            self.activation = kwargs['activation']
                        
        # select layer sizes and scale
        N = 1; M = 1;
        U = 10;
        self.layer_sizes = [N,U,M]
        self.scale = 0.1
        if 'layer_sizes' in kwargs:
            self.layer_sizes = kwargs['layer_sizes']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    # create initial weights for arbitrary feedforward network
    def initializer(self):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(self.layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = self.layer_sizes[k]
            U_k_plus_1 = self.layer_sizes[k+1]

            # make weight matrix
            weight = self.scale*np.random.randn(U_k+1,U_k_plus_1)
            weights.append(weight)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init

    ####### feature transforms ######
    # a feature_transforms function for computing
    # U_L L layer perceptron units efficiently
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W in w:
            # compute inner product with current layer weights
            a = W[0] + np.dot(a.T, W[1:])

            # output of layer activation
            a = self.activation(a).T
        return a
    
    

class Normalizer:
    def __init__(self,x,name):
        normalizer = 0
        inverse_normalizer = 0
        if name == 'standard':
            # create normalizer
            self.normalizer, self.inverse_normalizer = self.standard_normalizer(x)
            
        elif name == 'sphere':
            # create normalizer
            self.normalizer, self.inverse_normalizer = self.PCA_sphereing(x)
        else:
            self.normalizer = lambda data: data
            self.inverse_normalizer = lambda data: data
            
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    # compute eigendecomposition of data covariance matrix for PCA transformation
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        d,V = np.linalg.eigh(Cov)
        return d,V

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
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((stds.shape))
            adjust[ind] = 1.0
            stds += adjust
        
        normalizer = lambda data: np.dot(V.T,data - x_means)/stds

        # create inverse normalizer
        inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

        # return normalizer 
        return normalizer,inverse_normalizer
    


class Setup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x = x
        self.y = y
        
        # make containers for all histories
        self.weight_histories = []
        self.cost_histories = []
        self.count_histories = []
        
    #### define feature transformation ####
    def choose_features(self,name,**kwargs): 
        ### select from pre-made feature transforms ###
        # multilayer perceptron #
        if name == 'multilayer_perceptron':
            transformer = MultilayerPerceptron(**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            self.layer_sizes = transformer.layer_sizes

    #### define normalizer ####
    def choose_normalizer(self,name):
        # produce normalizer / inverse normalizer
        s = Normalizer(self.x,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x)
        self.normalizer_name = name
     
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # pick cost based on user input
        funcs = CostFunction(name,self.x,self.y,self.feature_transforms,**kwargs)
        self.cost = funcs.cost
        self.model = funcs.model
        
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = CostFunction('twoclass_counter',self.x,self.y,self.feature_transforms,**kwargs)
            self.counter = funcs.cost
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = CostFunction('multiclass_counter',self.x,self.y,self.feature_transforms,**kwargs)
            self.counter = funcs.cost
        self.cost_name = name
            
    #### run optimization ####
    def fit(self,**kwargs):
        # basic parameters for gradient descent run (default algorithm)
        max_its = 500; alpha_choice = 10**(-1);
        self.w_init = self.initializer()
        optimizer = 'gradient descent'
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer']
        
        # set parameters by hand
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            self.alpha_choice = kwargs['alpha_choice']
            
        # batch size for gradient descent?
        self.num_pts = np.size(self.y)
        self.batch_size = np.size(self.y)
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']

        # optimize
        weight_history = []
        cost_history = []
        
        if optimizer == 'gradient descent':
            # run gradient descent
            weight_history,cost_history = gradient_descent(self.cost,self.alpha_choice,self.max_its,self.w_init,self.num_pts,self.batch_size)
            
        # store all new histories
        self.weight_histories.append(weight_history)
        self.cost_histories.append(cost_history)
        
        # if classification produce count history
        if self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            count_history = [self.counter(v) for v in weight_history]
            
            # store count history
            self.count_histories.append(count_history)
 
    #### plot histories ###
    def show_histories(self,**kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
            
        # if labels not in input argument, make blank labels
        labels = []
        for c in range(len(self.cost_histories)):
            labels.append('')
        if 'labels' in kwargs:
            labels = kwargs['labels']
        HistoryPlotter(self.cost_histories,self.count_histories,start,labels)
        
    