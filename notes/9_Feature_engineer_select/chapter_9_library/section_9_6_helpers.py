import copy
from inspect import signature

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func


#### optimizers ####
# minibatch gradient descent
def gradient_descent(g,w,x,y,alpha_choice,max_its,batch_size): 
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    num_train = y.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x,y,np.arange(num_train))]

    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))

    # over the line
    alpha = 0
    print ('grads')

    for k in range(max_its):             
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
            
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))

            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,x,y,batch_inds)
            grad_eval.shape = np.shape(w)

            # take descent step with momentum
            w = w - alpha*grad_eval

        # update training and validation cost
        train_cost = g_flat(w,x,y,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
    return w_hist,train_hist

# newtons method function
def newtons_method(g,w,x,y,max_its,**kwargs): 
    # flatten input funciton, in case it takes in matrices of weights
    g_flat, unflatten, w = flatten_func(g, w)
    
    # compute the gradient / hessian functions of our input function
    grad = value_and_grad(g_flat)
    hess = hessian(g_flat)
    
    # set numericxal stability parameter / regularization parameter
    epsilon = 10**(-7)
    if 'epsilon' in kwargs:
        epsilon = kwargs['epsilon']
    
    # record history
    num_train = y.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x,y,np.arange(num_train))]

    # over the line
    for k in range(max_its):   
        # evaluate the gradient, store current weights and cost function value
        cost_eval,grad_eval = grad(w,x,y,np.arange(num_train))

        # evaluate the hessian
        hess_eval = hess(w,x,y,np.arange(num_train))

        # reshape for numpy linalg functionality
        hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))

        # solve second order system system for weight update
        A = hess_eval + epsilon*np.eye(np.size(w))
        b = grad_eval
        w = np.linalg.lstsq(A,np.dot(A,w) - b)[0]

        #w = w - np.dot(np.linalg.pinv(hess_eval + epsilon*np.eye(np.size(w))),grad_eval)
            
        # update training and validation cost
        train_cost = g_flat(w,x,y,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
    
    return w_hist,train_hist

class CostSetup:
    def __init__(self,name):             
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
            
    ###### cost functions #####
    # set model
    def set_model(self,model):
        self.model = model
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum((self.model(x_p,w) - y_p)**2)
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum(np.abs(self.model(x_p,w) - y_p))
        return cost/float(np.size(y_p))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*(self.model(x_p,w)))))
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        return cost/float(np.size(y_p))
    
    # the counting cost function
    def counting_cost(self,w,x,y,iter):
        cost = np.sum(np.abs(np.sign(self.model(x,w)) - self.y))
        return cost 

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,x,y,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]

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
    def multiclass_softmax(self,w,x,y,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
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
    def multiclass_counting_cost(self,w,x,y,iter):            
        # pre-compute predictions on all points
        all_evals = self.model(x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(y - y_predict)))

        # return number of misclassifications
        return count
    
class NormalizerSetup:
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
 
    # standard normalization function - with nan checker / filler in-er
    def standard_normalizer(self,x):    
        # compute the mean and standard deviation of the input
        x_means = np.nanmean(x,axis = 1)[:,np.newaxis]
        x_stds = np.nanstd(x,axis = 1)[:,np.newaxis]   

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # fill in any nan values with means 
        ind = np.argwhere(np.isnan(x) == True)
        for i in ind:
            x[i[0],i[1]] = x_means[i[0]]

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
    

class BoostSetup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x_orig = x
        self.y_orig = y
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_count_histories = []
        self.valid_cost_histories = []
        self.valid_count_histories = []

    #### define normalizer ####
    def choose_normalizer(self,name):       
        # produce normalizer / inverse normalizer
        s = NormalizerSetup(self.x_orig,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x_orig)
        self.normalizer_name = name
        
        # produce normalizer / inverse normalizer
        s = NormalizerSetup(self.y_orig,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.y = self.normalizer(self.y_orig)
  
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # create cost on entire dataset
        self.cost = CostSetup(name)
                
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = CostSetup('twoclass_counter')
            self.counter = funcs.cost
            
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = CostSetup('multiclass_counter')
            self.counter = funcs.cost
            
        self.cost_name = name
            
    #### setup optimization ####
    def choose_optimizer(self,optimizer_name,**kwargs):
        # general params for optimizers
        max_its = 500; 
        alpha_choice = 10**(-1);
        epsilon = 10**(-10)
        
        # set parameters by hand
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            alpha_choice = kwargs['alpha_choice']
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            
        # batch size for gradient descent?
        self.w = 0.0*np.random.randn(self.x.shape[0] + 1,1)
        num_pts = np.size(self.y)
        batch_size = np.size(self.y)
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        
        # run gradient descent
        if optimizer_name == 'gradient_descent':
            self.optimizer = lambda cost,x,w: gradient_descent(cost,w,x,self.y,alpha_choice,max_its,batch_size)
        
        if optimizer_name == 'newtons_method':
            self.optimizer = lambda cost,x,w: newtons_method(cost,w,x,self.y,max_its,epsilon=epsilon)
       
    ### boost it ###
    def boost(self,**kwargs):
        # choose number of rounds
        num_rounds = self.x.shape[0]
        if 'num_rounds' in kwargs:
            num_rounds = min(kwargs['num_rounds'],self.x.shape[0])
              
        # reset initialization
        self.w = 0.0*np.random.randn(self.x.shape[0] + 1,1)

        # container for models and cost function histories
        self.models = []
        self.cost_vals = []
        self.weight_vals = []
        
        # tune bias
        model_0 = lambda x,w: w
        self.cost.set_model(model_0)
        w_hist,c_hist = self.optimizer(self.cost.cost,self.x,self.w[0])

        # determine smallest cost value attained
        ind = np.argmin(c_hist)
        self.w[0] = w_hist[ind][0]
        self.cost_vals.append(c_hist[ind])
        self.weight_vals.append(self.w[0])
        
        # lock in model_0 value
        model_0 = copy.deepcopy(self.w[0])
        
        self.models.append(model_0)
        cost_val = c_hist[ind]
        
        # loop over feature-touching weights and update one at a time
        model = lambda x,w: x*w
        model_m = lambda x,w: self.models[0] + model(x,w)
        
        # index sets to keep track of which feature-touching weights have been used
        # thus far
        used = [0]
        unused = {i for i in range(1,self.x.shape[0]+1)}
        
        for i in range(num_rounds):
            # loop over unused indices and try out each remaining corresponding weight
            best_weight = 0
            best_cost = np.inf
            best_ind = 0
            for n in unused:
                # construct model to test
                current_model = lambda x,w: self.models[-1] + model(x,w)

                # load in current model
                self.cost.set_model(current_model)
                w_hist,c_hist = self.optimizer(self.cost.cost,self.x[n-1,:][np.newaxis,:],self.w[n])

                # determine smallest cost value attained
                ind = np.argmin(c_hist)            
                weight = w_hist[ind]
                cost_val = c_hist[ind]

                # update smallest cost val / associated weight
                if cost_val < best_cost:
                    best_weight = weight
                    best_cost = cost_val
                    best_ind = n

            # after sweeping through and computing minimum for all subproblems
            # update the best weight value
            self.w[best_ind] = best_weight
            self.cost_vals.append(best_cost)
            self.weight_vals.append(self.w[best_ind])
            
            # fix next model
            model_m = self.models[-1] + self.x[best_ind-1,:][np.newaxis,:]*self.w[best_ind]
            self.models.append(model_m)

            # update current model
            model_m = lambda x,w: self.models[-1](x) + model(x,w)

            # remove best index from unused set, add to used set
            unused -= {best_ind}
            used.append(best_ind)
            
        # make universals
        self.used = used
        
    #### plotting functionality ###
    def plot_history(self):
        # colors for plotting
        colors = [[0,0.7,1],[1,0.8,0.5]]

        # initialize figure
        fig = plt.figure(figsize = (10,5.5))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(3, 1,height_ratios = [1,0.1,1]) 
        ax = plt.subplot(gs[0]); 
        
        ### plot history val ###
        ax.plot(self.cost_vals,linewidth = 2,color = colors[0]) 
        ax.scatter(np.arange(len(self.cost_vals)).flatten(),self.cost_vals,s = 70,color = colors[0],edgecolor = 'k',linewidth = 1,zorder = 5) 

        # change tick labels to used
        ax.set_xticks(np.arange(len(self.cost_vals)))
        ax.set_xticklabels(self.used)
        
        # clean up panel / axes labels
        xlabel = 'weight index'
        ylabel = 'cost value'
        title = 'cost value at each round of boosting'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 25)
        ax.set_title(title,fontsize = 16)
        
        # histogram plot of each non-bias weight
        ax.axhline(c='k',zorder = 2)
            
        ### make bar plot ###
        ax = plt.subplot(gs[1]); ax.axis('off')
        ax = plt.subplot(gs[2]); 
        ax.bar(np.arange(1,len(self.weight_vals)), [w[0] for w in self.weight_vals[1:]], color='k', alpha=0.5)
        ax.axhline(c='k',zorder = 2)
        
        # change tick labels to used
        ax.set_xticks(np.arange(1,len(self.cost_vals)))
        ax.set_xticklabels(self.used[1:])
        
        # dress panel
        xlabel = 'weight index'
        ylabel = 'weight value'
        title = 'weight values learned by boosting'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 25)
        ax.set_title(title,fontsize = 15)
        
    # static graphics
    def plot_regress(self,id1,labels):
        # initialize figure
        fig = plt.figure(figsize = (9,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # scatter plot
        ax.scatter(self.x[id1-1,:],self.y,color = 'k',edgecolor = 'w',s = 30)
    
        # dress panel
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])