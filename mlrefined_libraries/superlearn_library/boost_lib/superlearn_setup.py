import autograd.numpy as np
from . import optimizers 
from . import cost_functions
from . import normalizers
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
        
        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.y_orig,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.y = self.normalizer(self.y_orig)
  
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # create cost on entire dataset
        self.cost = cost_functions.Setup(name)
                
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = cost_functions.Setup('twoclass_counter')
            self.counter = funcs.cost
            
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = cost_functions.Setup('multiclass_counter')
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
            self.optimizer = lambda cost,x,w: optimizers.gradient_descent(cost,w,x,self.y,alpha_choice,max_its,batch_size)
        
        if optimizer_name == 'newtons_method':
            self.optimizer = lambda cost,x,w: optimizers.newtons_method(cost,w,x,self.y,max_its,epsilon=epsilon)
       
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