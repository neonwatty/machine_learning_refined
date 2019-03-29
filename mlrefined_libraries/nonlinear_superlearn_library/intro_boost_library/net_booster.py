import autograd.numpy as np
from . import optimizers 
from . import cost_functions
from . import normalizers
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output
import time

class Setup:
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
        s = normalizers.Setup(self.x_orig,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x_orig)
        self.normalizer_name = name
        self.y = self.y_orig
        
    #### split into training / validation sets ####    
    def make_train_valid_split(self,train_portion):
        # translate desired training portion into exact indecies
        r = np.random.permutation(self.x.shape[1])
        train_num = int(np.round(train_portion*len(r)))
        self.train_inds = r[:train_num]
        self.valid_inds = r[train_num:]
        
        # define training and validation sets
        self.x_train = self.x[:,self.train_inds]
        self.x_valid = self.x[:,self.valid_inds]
        
        self.y_train = self.y[:,self.train_inds]
        self.y_valid = self.y[:,self.valid_inds]   
  
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # create cost on entire dataset
        self.cost = cost_functions.Setup(name)
                
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            self.counter = cost_functions.Setup('twoclass_counter')
            
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            self.counter = cost_functions.Setup('multiclass_counter')
            
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
            self.optimizer = lambda cost,x,y,w: optimizers.gradient_descent(cost,w,x,y,alpha_choice,max_its,batch_size)
        
        if optimizer_name == 'newtons_method':
            self.optimizer = lambda cost,x,y,w: optimizers.newtons_method(cost,w,x,y,max_its,epsilon=epsilon)

    # define activation
    def choose_activation(self,activation):
        if activation == 'tanh':
            self.activation = lambda data: np.tanh(data)
        elif activation == 'relu':
            self.activation = lambda data: np.maximum(0,data)
      
    
    # fully evaluate our network features using the tensor of weights in w
    def perceptron(self,a, w):    
        # compute inner product with current layer weights
        a = w[0][0] + np.dot(a.T, w[0][1:])

        # output of layer activation
        a = self.activation(a).T
        
        # final linear combo 
        a = w[1][0] + np.dot(a.T,w[1][1:])
        return a.T
        
    ### boost it ###
    def boost(self,num_rounds,**kwargs): 
        verbose = True
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        
        # container for models and cost function histories
        self.best_steps = []
        self.train_cost_vals = []
        self.valid_cost_vals = []
        self.models = []
        
        # tune bias
        model_0 = lambda x,w: w*np.ones((1,x.shape[1]))
        self.cost.set_model(model_0)
        w = 0.1*np.random.randn(1)
        w_hist,c_hist = self.optimizer(self.cost.cost,self.x_train,self.y_train,w)
        
        # determine smallest cost value attained
        ind = np.argmin(c_hist)
        w_best = w_hist[ind]

        # lock in model_0 value
        model = lambda x,w=w_best: model_0(x,w)
        self.best_steps.append(copy.deepcopy(model))
        self.models.append(copy.deepcopy(model))
        model = lambda x,steps=self.best_steps: np.sum([v(x) for v in steps],axis=0)
        
        train_cost_val = c_hist[ind]
        self.train_cost_vals.append(copy.deepcopy(train_cost_val))

        if self.y_valid.size > 0:
            valid_cost_val = self.cost.cost(w_best,self.x_valid,self.y_valid,np.arange(len(self.y_valid)))
            self.valid_cost_vals.append(copy.deepcopy(valid_cost_val))
        
        # pluck counter
        if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            self.train_count_vals = []
            self.valid_count_vals = []     
            
        # pluck counter
        if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            self.counter.set_model(model)

            train_count = self.counter.cost(self.x_train,self.y_train)
            self.train_count_vals.append(train_count)
            
            if self.y_valid.size > 0:
                valid_count = self.counter.cost(self.x_valid,self.y_valid)
                self.valid_count_vals.append(valid_count)   

        # boost rounds
        for i in range(num_rounds):     
            if verbose: 
                print ('starting round ' + str(i+1) + ' of ' + str(num_rounds) + ' of boosting')
           
            # initialize weights
            scale = 0.1
            U = 1
            w = [scale*np.random.randn(self.x.shape[0] + 1,U), scale*np.random.randn(2,U)]
    
            # construct model to test
            next_unit = lambda x,w: self.perceptron(x,w)
            current_model = lambda x,w: model(x) + next_unit(x,w)
        
            # load in current model
            self.cost.set_model(current_model)
            w_hist,c_hist = self.optimizer(self.cost.cost,self.x_train,self.y_train,w)
            
            # determine smallest cost value attained
            ind = np.argmin(c_hist)            
            w_best = w_hist[ind]
            best_train_cost = c_hist[ind]
            self.train_cost_vals.append(copy.deepcopy(best_train_cost))


            if self.y_valid.size > 0:
               best_valid_cost = self.cost.cost(w_best,self.x_valid,self.y_valid,np.arange(len(self.y_valid)))
               self.valid_cost_vals.append(copy.deepcopy(best_valid_cost))

            # pluck counter
            if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
                self.counter.set_model(model)

                train_count = self.counter.cost(self.x_train,self.y_train)
                self.train_count_vals.append(train_count)

                self.valid_count_vals.append(valid_count)  
                valid_count = self.counter.cost(self.x_valid,self.y_valid)
 
            
            # best_perceptron = lambda x,w=w_best: np.dot(self.perceptron(x,w[0]).T,w[1]).T 
            best_perceptron = lambda x,w=w_best: next_unit(x,w)
            self.best_steps.append(copy.deepcopy(best_perceptron))
            
            # fix next model
            model = lambda x,steps=self.best_steps: np.sum([v(x) for v in steps],axis=0)
            self.models.append(copy.deepcopy(model))
  
        if verbose:
            print ('boosting complete!')
            time.sleep(1.5)
            clear_output()
        
    #### plotting functionality ###
    def plot_history(self):
        # colors for plotting
        colors = [[0,0.7,1],[1,0.8,0.5]]

        # initialize figure
        fig = plt.figure(figsize = (9,4))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        ### plot history val ###
        ax.plot(self.train_cost_vals,linewidth = 2,color = colors[0]) 
        ax.plot(self.valid_cost_vals,linewidth = 2,color = colors[1]) 
        
        # clean up panel / axes labels
        xlabel = 'boosting round'
        ylabel = 'cost value'
        title = 'cost value at each round of boosting'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 25)
        ax.set_title(title,fontsize = 16)
        
        # histogram plot of each non-bias weight
        ax.axhline(c='k',zorder = 2)
        
         
    #### plotting functionality ###
    def plot_misclass_history(self):     
        # colors for plotting
        colors = [[0,0.7,1],[1,0.8,0.5]]

        # initialize figure
        fig = plt.figure(figsize = (9,4))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        ### plot history val ###
        ax.plot(self.train_count_vals,linewidth = 2,color = colors[0]) 
        ax.plot(self.valid_count_vals,linewidth = 2,color = colors[1]) 
        
        #ax.scatter(np.arange(len(self.cost_vals)).flatten(),self.cost_vals,s = 70,color = colors[0],edgecolor = 'k',linewidth = 1,zorder = 5) 

        # change tick labels to used
        #ax.set_xticks(np.arange(len(self.cost_vals)))
        #ax.set_xticklabels(self.used)
        
        # clean up panel / axes labels
        xlabel = 'boosting round'
        ylabel = 'number of misclassifications'
        title = 'misclassifications at each round of boosting'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 25)
        ax.set_title(title,fontsize = 16)
        
        # histogram plot of each non-bias weight
        ax.axhline(c='k',zorder = 2)       
        