import autograd.numpy as np
from . import unsuper_optimizers 
from . import unsuper_cost_functions
from . import normalizers
from . import multilayer_perceptron
from . import history_plotters

class Setup:
    def __init__(self,X,**kwargs):
        # link in data
        self.x = X
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_accuracy_histories = []
        self.val_cost_histories = []
        self.val_accuracy_histories = []
        self.train_costs = []
        self.train_counts = []
        self.val_costs = []
        self.val_counts = []
        
    #### define preprocessing steps ####
    def preprocessing_steps(self,**kwargs):        
        ### produce / use data normalizer ###
        normalizer_name = 'standard'
        if 'normalizer_name' in kwargs:
            normalizer_name = kwargs['normalizer_name']
        self.normalizer_name = normalizer_name

        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.x,normalizer_name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x)
        
    #### split data into training and validation sets ####
    def make_train_val_split(self,train_portion):
        # translate desired training portion into exact indecies
        self.train_portion = train_portion
        r = np.random.permutation(self.x.shape[1])
        train_num = int(np.round(train_portion*len(r)))
        self.train_inds = r[:train_num]
        self.val_inds = r[train_num:]
        
        # define training and testing sets
        self.x_train = self.x[:,self.train_inds]
        self.x_val = self.x[:,self.val_inds]
        
    #### define encoder ####
    def choose_encoder(self,**kwargs):         
        feature_name = 'multilayer_perceptron'
        if 'name' in kwargs:
            feature_name = kwargs['feature_name']
        
        transformer = 0
        if feature_name == 'multilayer_perceptron':
            transformer = multilayer_perceptron.Setup(**kwargs)
        elif feature_name == 'multilayer_perceptron_batch_normalized':
            transformer = multilayer_perceptron_batch_normalized.Setup(**kwargs)
            
        self.feature_transforms = transformer.feature_transforms
        self.initializer_1 = transformer.initializer
     
    # form decoder
    def choose_decoder(self,**kwargs):         
        feature_name = 'multilayer_perceptron'
        if 'name' in kwargs:
            feature_name = kwargs['feature_name']
           
        transformer = 0
        if feature_name == 'multilayer_perceptron':
            transformer = multilayer_perceptron.Setup(**kwargs)
        elif feature_name == 'multilayer_perceptron_batch_normalized':
            transformer = multilayer_perceptron_batch_normalized.Setup(**kwargs)
        
        self.feature_transforms_2 = transformer.feature_transforms
        self.initializer_2 = transformer.initializer
     
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # pick cost based on user input
        self.cost_object = unsuper_cost_functions.Setup(name,**kwargs)
                
        ### with feature transformation constructed, pass on to cost function ###
        self.cost_object.define_encoder_decoder(self.feature_transforms,self.feature_transforms_2)
        self.cost = self.cost_object.cost
        self.cost_name = name
        self.encoder = self.cost_object.encoder
        self.decoder = self.cost_object.decoder
            
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

        # batch size for gradient descent?
        self.train_num = np.shape(self.x_train)[1]
        self.val_num = np.shape(self.x_val)[1]
        self.batch_size = np.shape(self.x_train)[1]
        if 'batch_size' in kwargs:
            self.batch_size = min(kwargs['batch_size'],self.batch_size)
        
        # verbose or not
        verbose = True
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        
        # run gradient descent
        weight_history,train_cost_history,val_cost_history = unsuper_optimizers.gradient_descent(self.cost,self.w_init,self.x_train,self.x_val,self.alpha_choice,self.max_its,self.batch_size,verbose=verbose)
        
        # store all new histories
        self.weight_histories.append(weight_history)
        self.train_cost_histories.append(train_cost_history)
        self.val_cost_histories.append(val_cost_history)
        
    #### plot histories ###
    def show_histories(self,**kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
        if self.train_portion == 1:
            self.val_cost_histories = [[] for s in range(len(self.val_cost_histories))]
            self.val_accuracy_histories = [[] for s in range(len(self.val_accuracy_histories))]
        history_plotters.Setup(self.train_cost_histories,self.train_accuracy_histories,self.val_cost_histories,self.val_accuracy_histories,start)
        
    