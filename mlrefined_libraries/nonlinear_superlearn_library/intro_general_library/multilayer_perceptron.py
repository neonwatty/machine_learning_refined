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

    # fully evaluate our network features using the tensor of weights in w
    def feature_transforms(self,a, w):    
        # loop through each layer matrix
        for W in w:
            #  pad with ones (to compactly take care of bias) for next layer computation        
            o = np.ones((1,np.shape(a)[1]))
            a = np.vstack((o,a))

            # compute inner product with current layer weights
            a = np.dot(a.T, W).T

            # output of layer activation
            a = self.activation(a)
        return a