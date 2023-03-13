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
        else: # user-defined activation
            self.activation = kwargs['activation']
                        
        # select layer sizes and scale
        N = 1; M = 1;
        U = 10;
        self.layer_sizes = [N,U,M]
        self.scale = 0.1
        if 'layer_sizes' in kwargs:
            self.layer_sizes = kwargs['layer_sizes']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    # create initial weights for arbitrary feedforward network
    def initializer(self):
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

    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # return normalizer 
        return normalizer
    

    ####### feature transforms ######
    # a feature_transforms function for computing
    # U_L L layer perceptron units efficiently
    # -- batch normalized
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W in w:
            # compute inner product with current layer weights
            a = W[0] + np.dot(a.T, W[1:])

            # output of layer activation
            a = self.activation(a).T

            # perform standard normalization to the activation outputs
            normalizer = self.standard_normalizer(a)
            a = normalizer(a)
        return a