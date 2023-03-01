import autograd.numpy as np
from . import optimizers 
from . import cost_functions
from . import normalizers
from . import multilayer_perceptron
from . import history_plotters

class Setup:
    def __init__(self,X,**kwargs):
        # link in data
        self.x = X
        
        # make containers for all histories
        self.weight_histories = []
        self.cost_histories = []
        self.count_histories = []
        
    #### define feature transformation ####
    def choose_encoder(self,**kwargs): 
        # select from pre-made feature transforms
        # form encoder
        transformer = multilayer_perceptron.Setup(**kwargs)
        self.feature_transforms = transformer.feature_transforms
        self.initializer_1 = transformer.initializer
        self.layer_sizes_encoder = transformer.layer_sizes
        
    def choose_decoder(self,**kwargs): 
        # form decoder
        transformer = multilayer_perceptron.Setup(**kwargs)
        self.feature_transforms_2 = transformer.feature_transforms
        self.initializer_2 = transformer.initializer
        self.layer_sizes_decoder = transformer.layer_sizes

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
        funcs = cost_functions.Setup(name,self.x,[],self.feature_transforms,feature_transforms_2 = self.feature_transforms_2,**kwargs)
        self.cost = funcs.cost
        self.encoder = funcs.encoder
        self.decoder = funcs.decoder
        self.cost_name = name
            
    #### run optimization ####
    def fit(self,**kwargs):
        # basic parameters for gradient descent run (default algorithm)
        max_its = 500; alpha_choice = 10**(-1);
        self.w_init_1 = self.initializer_1()
        self.w_init_2 = self.initializer_2()
        self.w_init = [self.w_init_1,self.w_init_2]
        
        # set parameters by hand
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            self.alpha_choice = kwargs['alpha_choice']
        if 'w' in kwargs:
            self.w_init = kwargs['w']

        # run gradient descent
        self.weight_history, self.cost_history = optimizers.gradient_descent(self.cost,self.alpha_choice,self.max_its,self.w_init)
        
         # store all new histories
        self.weight_histories.append(self.weight_history)
        self.cost_histories.append(self.cost_history)
        
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
        
    