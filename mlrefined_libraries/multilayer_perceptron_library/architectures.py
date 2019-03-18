# import autograd functionality
from autograd import grad as compute_grad  
import autograd.numpy as np
import copy

class Setup:
    '''
    Normalized multilayer perceptron / feedforward network architectures
    '''
 
    def choose_architecture(self,activation_name):
        self.activation_name = activation_name
        
        # set activation
        if activation_name == 'relu':
            self.activation = self.relu
        if activation_name == 'maxout':
            self.activation = self.maxout
        if activation_name == 'tanh':
            self.activation = self.tanh
        if activation_name == 'linear':
            self.activation = self.linear
            
        # set architecture and initializer (basically just a switch for maxout vs others)
        self.training_architecture = self.compute_general_network_features
        self.initializer = self.initialize_general_network_weights
        self.testing_architecture = self.compute_network_features_testing
        if self.activation_name == 'maxout':
            self.training_architecture = self.compute_maxout_network_features            
            self.initializer = self.initialize_maxout_network_weights   
            self.testing_architecture = self.compute_maxout_network_features_testing
            
    # our normalization function
    def normalize(self,data,data_mean,data_std):
        normalized_data = (data - data_mean)/(data_std + 10**(-5))
        return normalized_data

    ########## architectures ##########
    def compute_general_network_features(self,x, inner_weights):        
        # pad data with ones to deal with bias
        o = np.ones((np.shape(x)[0],1))
        a_padded = np.concatenate((o,x),axis = 1)

        # loop through weights and update each layer of the network
        for W in inner_weights:            
            # output of layer activation
            a = self.activation(np.dot(a_padded,W))

            ### normalize output of activation
            # compute the mean and standard deviation of the activation output distributions
            a_means = np.mean(a,axis = 0)
            a_stds = np.std(a,axis = 0)

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)

        return a_padded

    def compute_maxout_network_features(self,x, inner_weights):
        # pad data with ones to deal with bias
        o = np.ones((np.shape(x)[0],1))
        a_padded = np.concatenate((o,x),axis = 1)

        # loop through weights and update each layer of the network
        for W1,W2 in inner_weights:                                 
            # output of layer activation  
            a = self.activation(np.dot(a_padded,W1),np.dot(a_padded,W2))  

            ### normalize output of activation
            # compute the mean and standard deviation of the activation output distributions
            a_means = np.mean(a,axis = 0)
            a_stds = np.std(a,axis = 0)

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)

        return a_padded

    ########## test versions of the architecture to extract stats ##########
    def compute_network_features_testing(self,x, inner_weights,stats):
        '''
        An adjusted normalized architecture compute function that collects network statistics as the training data
        passes through each layer, and applies them to properly normalize test data.
        '''
        # are you using this to compute stats on training data (stats empty) or to normalize testing data (stats not empty)
        switch =  'testing'
        if np.size(stats) == 0:
            switch = 'training'

        # pad data with ones to deal with bias
        o = np.ones((np.shape(x)[0],1))
        a_padded = np.concatenate((o,x),axis = 1)

        # loop through weights and update each layer of the network
        c = 0
        for W in inner_weights:
            # output of layer activation
            a = self.activation(np.dot(a_padded,W))

            ### normalize output of activation
            a_means = 0
            a_stds = 0
            if switch == 'training':
                # compute the mean and standard deviation of the activation output distributions
                a_means = np.mean(a,axis = 0)
                a_stds = np.std(a,axis = 0)
                stats.append([a_means,a_stds])
            elif switch == 'testing':
                a_means = stats[c][0]
                a_stds = stats[c][1]

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)
            c+=1

        return a_padded,stats

    def compute_maxout_network_features_testing(self,x,inner_weights,stats):
        '''
        An adjusted normalized architecture compute function that collects network statistics as the training data
        passes through each layer, and applies them to properly normalize test data.
        '''
        # are you using this to compute stats on training data (stats empty) or to normalize testing data (stats not empty)
        switch =  'testing'
        if np.size(stats) == 0:
            switch = 'training'

        # pad data with ones to deal with bias
        o = np.ones((np.shape(x)[0],1))
        a_padded = np.concatenate((o,x),axis = 1)

        # loop through weights and update each layer of the network
        c = 0
        for W1,W2 in inner_weights:                                  
            # output of layer activation
            a = self.activation(np.dot(a_padded,W1),np.dot(a_padded,W2))     

            ### normalize output of activation
            a_means = 0
            a_stds = 0
            if switch == 'training':
                # compute the mean and standard deviation of the activation output distributions
                a_means = np.mean(a,axis = 0)
                a_stds = np.std(a,axis = 0)
                stats.append([a_means,a_stds])
            elif switch == 'testing':
                a_means = stats[c][0]
                a_stds = stats[c][1]

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)
            c+=1

        return a_padded,stats

    ########## weight initializers ##########
    # create initial weights for arbitrary feedforward network
    def initialize_general_network_weights(self,layer_sizes,scale):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]

            # make weight matrix
            weight = scale*np.random.randn(U_k + 1,U_k_plus_1)
            weights.append(weight)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init[1] = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init
    
    # create initial weights for maxout feedforward network
    def initialize_maxout_network_weights(self,layer_sizes,scale):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]

            # make weight matrix
            weight1 = scale*np.random.randn(U_k + 1,U_k_plus_1)

            # add second matrix for inner weights
            if k < len(layer_sizes)-2:
                weight2 = scale*np.random.randn(U_k + 1,U_k_plus_1)
                weights.append([weight1,weight2])
            else:
                weights.append(weight1)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init
    
    ########## activation functions ##########
    def maxout(self,t1,t2):
        # maxout activation
        f = np.maximum(t1,t2)
        return f
    
    def relu(self,t):
        # relu activation
        f = np.maximum(0,t)
        return f    
    
    def tanh(self,t):
        # tanh activation
        f = np.tanh(t)
        return f    
    
    def linear(self,t):
        # linear activation
        f = t
        return f 