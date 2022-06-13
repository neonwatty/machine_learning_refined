# clear display
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np

# aome basic libraries
import math
import time
import copy

# plotting functionality
import matplotlib.pyplot as plt
from matplotlib import gridspec

##### import network functionality #####
from . import optimizers
from . import cost_functions
from . import architectures

class Network:
    '''
    Normalized multilayer perceptron / feedforward network learner
    '''    
    
    ###### load in training/testing data ######
    def input_data(self,train_data,test_data,normalize):
        self.train_data = train_data
        self.x_train = train_data[:,:-1]
        self.y_train = train_data[:,-1:]
        
        # normalize training data?
        if normalize == True:
            # training data
            self.x_means = np.mean(self.x_train,axis = 0)
            self.x_stds = np.std(self.x_train,axis = 0)
            self.x_train = self.normalize(self.x_train,self.x_means,self.x_stds)
                            
        # test data included?
        if np.size(test_data) > 0:
            self.test_data = test_data
            self.x_test = test_data[:,:-1]
            self.y_test = test_data[:,-1:]
            self.x_test = self.normalize(self.x_test,self.x_means,self.x_stds)  
        else:
            self.test_data = []
            self.x_test = []
            self.y_test = []
            
            
    # our normalization function
    def normalize(self,data,data_mean,data_std):
        normalized_data = (data - data_mean)/(data_std + 10**(-5))
        return normalized_data          
    
    ###### setup architecture ######
    def architecture_settings(self,activation_name,layer_sizes):
        # create instance of architectures
        self.architectures = architectures.Setup()
        
        # setup architecture
        self.activation_name = activation_name
        self.layer_sizes = layer_sizes
        self.architectures.choose_architecture(activation_name)
        
    ###### chose cost #####
    def choose_cost(self,cost_name):
        self.cost_name = cost_name
            
        # if cost is multiclass softmax then flatten output, this is done because the compact multiclass softmax 
        # written with broadcasting with y, and necessary for y to be flattened for this
        if self.cost_name == 'multiclass_softmax':
            self.y_train = np.asarray([int(v) for v in self.y_train])
            self.y_train = self.y_train.flatten()
            
            self.y_test = np.asarray([int(v) for v in self.y_test])
            self.y_test = self.y_test.flatten()
        
        # create instance of cost functions
        cost_function = cost_functions.Setup()
        
        # determine the right predict function
        # create instance of cost
        cost_function.choose_cost(cost_name,self.predict_training,self.x_train,self.y_train)
        self.training_cost = cost_function.cost
    
    ########## predict functions ##########
    # predict for training
    def predict_training(self,x,w):
        # feature trasnsformations
        f = self.architectures.training_architecture(x,w[0])

        # compute linear model
        vals = np.dot(f,w[1])
        return vals
    
    # predict for testing 
    def predict_testing(self,x,w):     
        # feature trasnsformations
        f,stats = self.architectures.testing_architecture(x,w[0],self.train_stats)

        # compute linear model
        vals = np.dot(f,w[1])
        return vals
        
    ###### setup optimizer ######
    def optimizer_settings(self,alpha,max_its,**kwargs):
        # generate initial weights
        scale = 0.1
        if 'scale' in kwargs:
            scale = kwargs['scale']
        self.w_init = self.architectures.initializer(self.layer_sizes,scale)
        
        # other settings
        self.alpha = alpha
        self.max_its =  max_its
        self.beta = 0
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        self.version = 'normalized'
        if 'version' in kwargs:
            self.version = kwargs['version']
        
        # create instance of optimizers
        self.opt = optimizers.Setup()
        
    def fit(self,**kwargs):
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            
        # run optimizer
        self.weight_history = self.opt.gradient_descent(self.training_cost,self.w_init,self.alpha,self.max_its,self.beta,self.version,verbose=verbose)
        
    ####### show cost function plots #######
    def compute_cost_plots(self): 
        # create instance of cost functions
        cost_function2 = cost_functions.Setup()
        
        # loop over weights in history and construct cost function plots for training and testing data
        self.train_cost_history = []
        self.test_cost_history = []
        self.training_stats = []
        
        # classification?  then record count history as well
        if self.cost_name == 'twoclass_softmax' or self.cost_name == 'multiclass_softmax':
            self.train_count_history = []
            self.test_count_history = []
        
        # loop over weights and record cost values
        for w in self.weight_history:
            # use testing architecture to gather stats on training data network normalization
            a_padded,self.train_stats = self.architectures.testing_architecture(self.x_train,w[0],[])
            self.training_stats.append(self.train_stats)

            # evalaute both training and testing data using testing predictor, which will normalize network
            # with respect to current weights and training data
            cost_function2.choose_cost(self.cost_name,self.predict_testing,self.x_train,self.y_train)
            testing_cost = cost_function2.cost   
            self.train_cost_history.append(testing_cost(w))
            
            # classification?  then record misclassification data too
            if self.cost_name == 'twoclass_softmax':
                cost_function2.choose_cost('twoclass_counter',self.predict_testing,self.x_train,self.y_train)
                testing_cost = cost_function2.cost   
                self.train_count_history.append(testing_cost(w))
            if self.cost_name == 'multiclass_softmax':
                cost_function2.choose_cost('multiclass_counter',self.predict_testing,self.x_train,self.y_train)
                testing_cost = cost_function2.cost   
                self.train_count_history.append(testing_cost(w))   
            
            # was test data included?  then compute error on this
            if np.size(self.test_data) > 0:
                cost_function2.choose_cost(self.cost_name,self.predict_testing,self.x_test,self.y_test)
                testing_cost = cost_function2.cost
                self.test_cost_history.append(testing_cost(w))
                
                # classification?  then record misclassification data too
                if self.cost_name == 'twoclass_softmax':
                    cost_function2.choose_cost('twoclass_counter',self.predict_testing,self.x_test,self.y_test)
                    testing_cost = cost_function2.cost   
                    self.test_count_history.append(testing_cost(w))
                if self.cost_name == 'multiclass_softmax':
                    cost_function2.choose_cost('multiclass_counter',self.predict_testing,self.x_test,self.y_test)
                    testing_cost = cost_function2.cost   
                    self.test_count_history.append(testing_cost(w))   
                
            
            ### if performing classification record number of micclassifications as well ###
            
    # plot cost function histories    
    def plot_histories(self,start):
        ### plot
        # initialize figure
        fig = plt.figure(figsize = (8,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        if self.cost_name == 'twoclass_softmax' or self.cost_name == 'multiclass_softmax':
            gs = gridspec.GridSpec(1, 2) 
            ax = plt.subplot(gs[0]); 
            ax2 = plt.subplot(gs[1]); 

        # now plot each, one per panel
        ax.plot(np.arange(start,len(self.train_cost_history),1),self.train_cost_history[start:],linewidth = 3*(0.8)**(1)) 
        ax.plot(np.arange(start,len(self.test_cost_history),1),self.test_cost_history[start:],linewidth = 3*(0.8)**(2)) 

        # label up
        ax.set_xlabel('iteration')
        ax.set_ylabel('cost function val')
        ax.set_title('cost function history')
        
        # test data included?  make sure to include correct labeling
        if np.size(self.test_data) > 0:
            ax.legend(['training','testing'],loc='upper right')
        else:
            ax.legend(['training'],loc='upper right')
            
        # classification?  then record count history as well
        if self.cost_name == 'twoclass_softmax' or self.cost_name == 'multiclass_softmax':
           # now plot each, one per panel
            ax2.plot(np.arange(start,len(self.train_count_history),1),self.train_count_history[start:],linewidth = 3*(0.8)**(1)) 
            ax2.plot(np.arange(start,len(self.test_count_history),1),self.test_count_history[start:],linewidth = 3*(0.8)**(2)) 
            
            # label up
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('misclassifications')
            ax2.set_title('misclassification history')
            
            # test data included?  make sure to include correct labeling
            if np.size(self.test_data) > 0:
                ax2.legend(['training','testing'],loc='upper right')
            else:
                ax2.legend(['training'],loc='upper right')
