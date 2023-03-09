import copy, time

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from IPython.display import clear_output
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

from inspect import signature

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
    
    
class CostSetup:
    def __init__(self,cost_name,reg_name):             
        ### make cost function choice ###
        # for regression
        if cost_name == 'least_squares':
            self.cost = self.least_squares
        if cost_name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # for two-class classification
        if cost_name == 'softmax':
            self.cost = self.softmax
        if cost_name == 'perceptron':
            self.cost = self.perceptron
        if cost_name == 'twoclass_counter':
            self.cost = self.counting_cost
            
        # for multiclass classification
        if cost_name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if cost_name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if cost_name == 'multiclass_counter':
            self.cost = self.multiclass_counting_cost
            
        # choose regularizer
        self.lam = 0
        if reg_name == 'L2':
            self.reg = self.L2
        if reg_name == 'L1':
            self.reg = self.L1
            
    ### regularizers ###
    def L1(self,w):
         return self.lam*np.sum(np.abs(w[1:]))
  
    def L2(self,w):
         return self.lam*np.sum((w[1:])**2)
        
    # set lambda value (regularization penalty)
    def set_lambda(self,lam):
        self.lam = lam
        
    ### setup model ###
    def model(self,x,w):
        a = w[0] + np.dot(x.T,w[1:])
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum((self.model(x_p,w) - y_p)**2)
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost
        cost = np.sum(np.abs(self.model(x_p,w) - y_p))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*(self.model(x_p,w)))))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        
        # add regularizer 
        cost += self.reg(w)
        
        # return average
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

        # add regularizer 
        cost += self.reg(w)
        
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

        # add regularizer 
        cost += self.reg(w)
        
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

class RegSetup:
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
        
        # normalize input 
        self.y = self.y_orig
  
    #### define cost function ####
    def choose_cost(self,cost_name,reg_name,**kwargs):
        # create cost on entire dataset
        self.cost = CostSetup(cost_name,reg_name)
                
        # if the cost function is a two-class classifier, build a counter too
        if cost_name == 'softmax' or cost_name == 'perceptron':
            funcs = CostSetup('twoclass_counter',reg_name)
            self.counter = funcs.cost
            
        if cost_name == 'multiclass_softmax' or cost_name == 'multiclass_perceptron':
            funcs = CostSetup('multiclass_counter',reg_name)
            self.counter = funcs.cost
            
        self.cost_name = cost_name
        self.reg_name = reg_name
            
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
       
    ### try-out various regularization params ###
    def tryout_lams(self,lams,**kwargs):
        # choose number of rounds
        self.lams = lams
        num_rounds = len(lams)

        # container for costs and weights 
        self.cost_vals = []
        self.weights = []
        
        # reset initialization
        self.w_init = 0.1*np.random.randn(self.x.shape[0] + 1,1)
            
        # loop over lams and try out each
        for i in range(num_rounds):        
            # set lambda
            lam = self.lams[i]
            self.cost.set_lambda(lam)
        
            # load in current model
            w_hist,c_hist = self.optimizer(self.cost.cost,self.x,self.w_init)
            
            # determine smallest cost value attained
            ind = np.argmin(c_hist)            
            weight = w_hist[ind]
            cost_val = c_hist[ind]
            self.weights.append(weight)
            self.cost_vals.append(cost_val)
            
        # determine best value of lamba from the above runs
        ind = np.argmin(self.cost_vals)
        self.best_lam = self.lams[ind]
        self.best_weights = self.weights[ind]
        
    # compare multiple l1 regularized runs
    def animate_lams(self,savepath,**kwargs):                   
        # initialize figure
        fig = plt.figure(figsize = (9,3))
        artist = fig
        gs = gridspec.GridSpec(1,1) 
        ax = plt.subplot(gs[0])
        
        ### run over all input lamdas ###
        # start animation
        num_frames = len(self.lams)
        print ('starting animation rendering...')
        def animate(k):            
            # clear panels
            ax.cla()
            lam = self.lams[k]
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # save lowest misclass weights
            w_best = self.weights[k][1:]
            
            # plot
            ax.axhline(c='k',zorder = 2)
            
            # make bar plot
            ax.bar(np.arange(1,len(w_best)+1).flatten(),np.array(w_best).flatten(), color='k', alpha=0.5)
                
            # dress panel
            title1 = r'$\lambda = ' + str(np.round(lam,2)) + '$' 
            costval = self.cost_vals[k][0]
            title2 = ', cost val = ' + str(np.round(costval,2))
            title = title1 + title2
            ax.set_title(title)
            ax.set_xlabel('learned weights')
            
            # change tick labels to used
            ax.set_xticks(np.arange(1,self.x.shape[0]+2))
            ax.set_xticklabels(np.arange(1,self.x.shape[0]+2))
            
            # clip viewing range
            ax.set_xlim([0.25,self.x.shape[0] + 0.75])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

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