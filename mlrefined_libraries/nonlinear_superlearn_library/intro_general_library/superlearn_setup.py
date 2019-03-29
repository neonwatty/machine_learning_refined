import autograd.numpy as np
from . import optimizers 
from . import cost_functions
from . import normalizers
from . import multilayer_perceptron
from . import multilayer_perceptron_batch_normalized
from . import stumps
from . import polys
from . import history_plotters

class Setup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x = x
        self.y = y
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_count_histories = []
        self.valid_cost_histories = []
        self.valid_count_histories = []
        
    #### define feature transformation ####
    def choose_features(self,name,**kwargs):         
        # multilayer perceptron #
        if name == 'multilayer_perceptron':
            self.transformer = multilayer_perceptron.Setup(**kwargs)
            self.feature_transforms = self.transformer.feature_transforms
            self.initializer = self.transformer.initializer
            self.layer_sizes = self.transformer.layer_sizes
            
        if name == 'multilayer_perceptron_batch_normalized':
            self.transformer = multilayer_perceptron_batch_normalized.Setup(**kwargs)
            self.feature_transforms = self.transformer.feature_transforms
            self.initializer = self.transformer.initializer
            self.layer_sizes = self.transformer.layer_sizes
            
        # stumps #
        if name == 'stumps':
            self.transformer = stumps.Setup(self.x,self.y,**kwargs)
            self.feature_transforms = self.transformer.feature_transforms
            self.initializer = self.transformer.initializer
            
        # polynomials #
        if name == 'polys':
            self.transformer = polys.Setup(self.x,self.y,**kwargs)
            self.feature_transforms = self.transformer.feature_transforms
            self.initializer = self.transformer.initializer
            self.degs = self.transformer.D
            
        # input a custom feature transformation
        if name == 'custom':
            self.feature_transforms = kwargs['feature_transforms']
            self.initializer = kwargs['initializer']
        self.feature_name = name

    #### define normalizer ####
    def choose_normalizer(self,name):
        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.x,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x)
        self.normalizer_name = name
       
    #### split data into training and validation sets ####
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
        self.lam = 0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']

        # create cost on entire dataset
        funcs = cost_functions.Setup(name,self.x,self.y,self.feature_transforms,**kwargs)
        self.full_cost = funcs.cost
        self.full_model = funcs.model

        # create training and testing cost functions
        funcs = cost_functions.Setup(name,self.x_train,self.y_train,self.feature_transforms,**kwargs)
        self.cost = funcs.cost
        self.model = funcs.model
        
        funcs = cost_functions.Setup(name,self.x_valid,self.y_valid,self.feature_transforms,**kwargs)
        self.valid_cost = funcs.cost
        
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = cost_functions.Setup('twoclass_counter',self.x_train,self.y_train,self.feature_transforms,**kwargs)
            self.counter = funcs.cost
            
            funcs = cost_functions.Setup('twoclass_counter',self.x_valid,self.y_valid,self.feature_transforms,**kwargs)
            self.valid_counter = funcs.cost
            
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = cost_functions.Setup('multiclass_counter',self.x_train,self.y_train,self.feature_transforms,**kwargs)
            self.counter = funcs.cost
            
            funcs = cost_functions.Setup('multiclass_counter',self.x_valid,self.y_valid,self.feature_transforms,**kwargs)
            self.valid_counter = funcs.cost
            
        self.cost_name = name
            
    #### run optimization ####
    def fit(self,**kwargs):
        # basic parameters for gradient descent run (default algorithm)
        max_its = 500; alpha_choice = 10**(-1);
        self.w_init = self.initializer()
        optimizer = 'gradient_descent'
        epsilon = 10**(-10)
        
        # set parameters by hand
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            self.alpha_choice = kwargs['alpha_choice']
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer']
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            
        # batch size for gradient descent?
        self.num_pts = np.size(self.y_train)
        self.batch_size = np.size(self.y_train)
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']

        # optimize
        weight_history = []
        
        # run gradient descent
        if optimizer == 'gradient_descent':
            weight_history = optimizers.gradient_descent(self.cost,self.alpha_choice,self.max_its,self.w_init,self.num_pts,self.batch_size)
        
        if optimizer == 'newtons_method':
            weight_history = optimizers.newtons_method(self.cost,self.max_its,self.w_init,self.num_pts,self.batch_size,epsilon = epsilon)
                    
        # compute training and testing cost histories
        train_cost_history = [self.cost(v,np.arange(np.size(self.y_train))) for v in weight_history]
        valid_cost_history = [self.valid_cost(v,np.arange(np.size(self.y_valid))) for v in weight_history]
        

        # store all new histories
        self.weight_histories.append(weight_history)
        self.train_cost_histories.append(train_cost_history)
        self.valid_cost_histories.append(valid_cost_history)
        
        # if classification produce count history
        if self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            train_count_history = [self.counter(v) for v in weight_history]
            valid_count_history = [self.valid_counter(v) for v in weight_history]

            # store count history
            self.train_count_histories.append(train_count_history)
            self.valid_count_histories.append(valid_count_history)
 
    #### plot histories ###
    def show_histories(self,**kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
        history_plotters.Setup(self.train_cost_histories,self.train_count_histories,self.valid_cost_histories,self.test_count_histories,start)
        
    #### for batch normalized multilayer architecture only - set normalizers to desired settings ####
    def fix_normalizers(self,w):
        ### re-set feature transformation ###        
        # fix normalization at each layer by passing data and specific weight through network
        self.feature_transforms(self.x,w);
        
        # re-assign feature transformation based on these settings
        self.feature_transforms_validation = self.transformer.feature_transforms_validation
        
        ### re-assign cost function (and counter) based on fixed architecture ###
        funcs = cost_functions.Setup(self.cost_name,self.x,self.y,self.feature_transforms_validation)
        self.model = funcs.model
    