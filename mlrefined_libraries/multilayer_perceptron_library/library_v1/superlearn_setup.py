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
        self.cost_histories = []
        self.count_histories = []
        
    #### define feature transformation ####
    def choose_features(self,name,**kwargs): 
        ### select from pre-made feature transforms ###
        # multilayer perceptron #
        if name == 'multilayer_perceptron':
            transformer = multilayer_perceptron.Setup(**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            self.layer_sizes = transformer.layer_sizes
            
        if name == 'multilayer_perceptron_batch_normalized':
            transformer = multilayer_perceptron_batch_normalized.Setup(**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            self.layer_sizes = transformer.layer_sizes
            
        # stumps #
        if name == 'stumps':
            transformer = stumps.Setup(self.x,self.y,**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            
        # polynomials #
        if name == 'polys':
            transformer = polys.Setup(self.x,self.y,**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            self.degs = transformer.D
            
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
     
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # pick cost based on user input
        funcs = cost_functions.Setup(name,self.x,self.y,self.feature_transforms,**kwargs)
        self.cost = funcs.cost
        self.model = funcs.model
        
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = cost_functions.Setup('twoclass_counter',self.x,self.y,self.feature_transforms,**kwargs)
            self.counter = funcs.cost
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = cost_functions.Setup('multiclass_counter',self.x,self.y,self.feature_transforms,**kwargs)
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
            weight_history,cost_history = optimizers.gradient_descent(self.cost,self.alpha_choice,self.max_its,self.w_init,self.num_pts,self.batch_size)
            
        if optimizer == 'RMSprop':
            weight_history,cost_history = optimizers.RMSprop(self.cost,self.alpha_choice,self.max_its,self.w_init,self.num_pts,self.batch_size)                
                
        if optimizer == 'newtons method':
            epsilon = 10**(-7)
            if 'epsilon' in kwargs:
                epsilon = kwargs['epsilon']
            weight_history,cost_history = optimizers.newtons_method(self.cost,self.max_its,self.w_init,self.num_pts,self.batch_size,epsilon = epsilon)

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
        history_plotters.Setup(self.cost_histories,self.count_histories,start,labels)
        
    