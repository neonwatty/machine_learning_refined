import autograd.numpy as np

class Setup:
    def __init__(self,**kwargs):        
        # set default values for layer sizes, activation, and scale
        activation = 'relu'

        # decide on these parameters via user input
        if 'activation' in kwargs:
            activation = kwargs['activation']

        # switches
        if activation == 'linear':
            self.activation = lambda data: data
        elif activation == 'tanh':
            self.activation = lambda data: np.tanh(data)
        elif activation == 'relu':
            self.activation = lambda data: np.maximum(0,data)
        elif activation == 'sinc':
            self.activation = lambda data: np.sinc(data)
        elif activation == 'sin':
            self.activation = lambda data: np.sin(data)
        elif activation == 'maxout':
            self.activation = lambda data1,data2: np.maximum(data1,data2)
                        
        # select layer sizes and scale
        self.layer_sizes = kwargs['layer_sizes']
        self.scale = 0.1
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
            
        # assign initializer / feature transforms function
        if activation == 'linear' or activation == 'tanh' or activation == 'relu' or activation == 'sinc' or activation == 'sin':
            self.initializer = self.standard_initializer
            self.feature_transforms = self.feature_transforms
        elif activation == 'maxout':
            self.initializer = self.maxout_initializer
            self.feature_transforms = self.maxout_feature_transforms

    ####### initializers ######
    # create initial weights for arbitrary feedforward network
    def standard_initializer(self):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(self.layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = self.layer_sizes[k]
            U_k_plus_1 = self.layer_sizes[k+1]

            # make weight matrix
            weight = self.scale*np.random.randn(U_k+1,U_k_plus_1)
            weights.append(weight)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init
    
    # create initial weights for arbitrary feedforward network
    def maxout_initializer(self):
        # container for entire weight tensor
        weights = []

        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(self.layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = self.layer_sizes[k]
            U_k_plus_1 = self.layer_sizes[k+1]

            # make weight matrix
            weight1 = self.scale*np.random.randn(U_k + 1,U_k_plus_1)

            # add second matrix for inner weights
            if k < len(self.layer_sizes)-2:
                weight2 = self.scale*np.random.randn(U_k + 1,U_k_plus_1)
                weights.append([weight1,weight2])
            else:
                weights.append(weight1)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],weights[-1]]

        return w_init

    ####### feature transforms ######
    # a feature_transforms function for computing
    # U_L L layer perceptron units efficiently
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W in w:
            # compute inner product with current layer weights
            a = W[0] + np.dot(a.T, W[1:])

            # output of layer activation
            a = self.activation(a).T
        return a
        
    # fully evaluate our network features using the tensor of weights in w
    def maxout_feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W1,W2 in w:
            # compute inner product with current layer weights
            a1 = W1[0] + np.dot(a.T, W1[1:])
            a2 = W2[0] + np.dot(a.T, W2[1:])

            # output of layer activation
            a = self.activation(a1,a2)
        return a