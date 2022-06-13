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
      
    
    ######## boosting demo with monomials  ########
    ### create prototype steps ###
    def create_monomials(self,D):
        N = self.x.shape[0]
        all_monos = []
        if N == 1:
            for d in range(D):
                mon = lambda x,deg = d: x**deg
                all_monos.append(copy.deepcopy(mon))

        if N == 2:
            degs = []
            for n in range(D):
                for m in range(D):
                    if n + m <= D:
                        degs.append([n,m])
            
            for deg in degs:
                mon = lambda x,n = deg[0], m = deg[1]: (x[0,:][np.newaxis,:]**n)*(x[1,:][np.newaxis,:]**m)
                all_monos.append(copy.deepcopy(mon))
        return all_monos
    
    ### boost it ###
    def boost(self,num_rounds,D,**kwargs):        
        # create monomials
        all_steps = self.create_monomials(D)        
        num_steps = len(all_steps)
        
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
        best_w = w_hist[ind][0]

        # lock in model_0 value
        model = lambda x,w=best_w: model_0(x,w)
        self.best_steps.append(copy.deepcopy(model))
        self.models.append(copy.deepcopy(model))
  
        train_cost_val = c_hist[ind]
        self.train_cost_vals.append(train_cost_val)
        valid_cost_val = self.cost.cost(best_w,self.x_valid,self.y_valid,np.arange(len(self.y_valid)))
        self.valid_cost_vals.append(valid_cost_val)

        # pluck counter
        if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            self.train_count_vals = []
            self.valid_count_vals = []     
            
        # pluck counter
        if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            self.counter.set_model(model)

            train_count = self.counter.cost(self.x_train,self.y_train)
            valid_count = self.counter.cost(self.x_valid,self.y_valid)

            self.train_count_vals.append(train_count)
            self.valid_count_vals.append(valid_count)   

        # index sets to keep track of which feature-touching weights have been used
        # thus far
        used = [0]
        unused = {i for i in range(1,num_steps+1)}
        for i in range(num_rounds):  
            print ('starting round ' + str(i+1) + ' of ' + str(num_rounds) + ' of boosting')
            # loop over unused indices and try out each remaining corresponding weight
            best_weight = 0
            best_train_cost = np.inf
            best_valid_cost = np.inf
            best_ind = 0

            for n in unused:
                # get current proto-step to test
                w = 0.1*np.random.randn(1)
                current_step = lambda x,w: w*all_steps[n-1](x)
                

                # construct model to test
                current_model = lambda x,w: model(x) + current_step(x,w)
                       
                # load in current model
                self.cost.set_model(current_model)
                w_hist,c_hist = self.optimizer(self.cost.cost,self.x_train,self.y_train,w)

                # determine smallest cost value attained
                ind = np.argmin(c_hist)         
                weight = w_hist[ind]
                train_cost_val = c_hist[ind]
                valid_cost_val = self.cost.cost(weight,self.x_valid,self.y_valid,np.arange(len(self.y_valid)))

                # update smallest cost val / associated weight
                if train_cost_val < best_train_cost:
                    best_w = weight
                    best_train_cost = train_cost_val
                    best_valid_cost = valid_cost_val
                    best_ind = n
            
            # after sweeping through and computing minimum for all subproblems
            # update the best weight value
            self.train_cost_vals.append(copy.deepcopy(best_train_cost))
            self.valid_cost_vals.append(copy.deepcopy(best_valid_cost))

            best_step = lambda x,w=best_w,ind=best_ind-1: w[0]*all_steps[ind](x)
            self.best_steps.append(copy.deepcopy(best_step))
            
            # fix next model
            model = lambda x,steps=self.best_steps: np.sum([v(x) for v in steps],axis = 0)
            self.models.append(copy.deepcopy(model))    

            # pluck counter
            if  self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
                self.counter.set_model(model)

                train_count = self.counter.cost(self.x_train,self.y_train)
                valid_count = self.counter.cost(self.x_valid,self.y_valid)
                
                self.train_count_vals.append(train_count)
                self.valid_count_vals.append(valid_count)   
            
            # remove best index from unused set, add to used set
            #unused -= {best_ind}
            # used.append(best_ind)
            
        # make universals
        self.used = used
        
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
        
        #ax.scatter(np.arange(len(self.cost_vals)).flatten(),self.cost_vals,s = 70,color = colors[0],edgecolor = 'k',linewidth = 1,zorder = 5) 

        # change tick labels to used
        #ax.set_xticks(np.arange(len(self.cost_vals)))
        #ax.set_xticklabels(self.used)
        
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
        
        # clean up panel / axes labels
        xlabel = 'boosting round'
        ylabel = 'number of misclassifications'
        title = 'misclassifications at each round of boosting'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 90,labelpad = 25)
        ax.set_title(title,fontsize = 16)
        
        # histogram plot of each non-bias weight
        ax.axhline(c='k',zorder = 2)