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
                        
        # get layer sizes
        layer_sizes = kwargs['layer_sizes']
        self.layer_sizes = layer_sizes
        self.scale = 0.1
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
            
        # assign initializer / feature transforms function
        if activation == 'linear' or activation == 'tanh' or activation == 'relu' or activation == 'sinc' or activation == 'sin':
            self.initializer = self.standard_initializer
            self.feature_transforms = self.standard_feature_transforms
            self.testing_feature_transforms = self.standard_feature_transforms_testing
        elif activation == 'maxout':
            self.initializer = self.maxout_initializer
            self.feature_transforms = self.maxout_feature_transforms
            self.testing_feature_transforms = self.maxout_feature_transforms_testing

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
    # a multilayer perceptron network, note the input w is a tensor of weights, with 
    # activation output normalization
    def standard_feature_transforms(self,a, w):    
        # loop through each layer matrix
        self.normalizers = []
        for W in w:
            # compute inner product with current layer weights
            a = W[0] + np.dot(a.T, W[1:])

            # output of layer activation
            a = self.activation(a).T

            # NEW - perform standard normalization to the activation outputs
            normalizer = self.standard_normalizer(a)
            a = normalizer(a)
            
            # store normalizer for testing data
            self.normalizers.append(normalizer)
        return a
    
    # a multilayer perceptron network, note the input w is a tensor of weights, with 
    # activation output normalization
    def maxout_feature_transforms(self,a, w):    
        # loop through each layer matrix
        self.normalizers = []
        for W1,W2 in w:
            # compute inner product with current layer weights
            a1 = W1[0][:,np.newaxis] + np.dot(a.T, W1[1:])
            a2 = W2[0][:,np.newaxis] + np.dot(a.T, W2[1:])

            # output of layer activation
            a = self.activation(a1,a2).T

            # NEW - perform standard normalization to the activation outputs
            normalizer = self.standard_normalizer(a)
            a = normalizer(a)
            
            # store normalizer for testing data
            self.normalizers.append(normalizer)
        return a
    
    #### testing side feature transforms ####
    # a copy of the batch normalized architecture that employs normalizers
    # at each layer based on statistics from training data and user-chosen
    # choice of weights w
    def standard_feature_transforms_testing(self,a, w):    
        # loop through each layer matrix
        c=0
        for W in w:
            #  pad with ones (to compactly take care of bias) for next layer computation        
            o = np.ones((1,np.shape(a)[1]))
            a = np.vstack((o,a))

            # compute linear combination of current layer units
            a = np.dot(a.T, W).T

            # pass through activation
            a = self.activation(a)

            # get normalizer for this layer tuned to training data
            normalizer = self.normalizers[c]
            a = normalizer(a)
            c+=1
        return a
    
    # a copy of the batch normalized architecture that employs normalizers
    # at each layer based on statistics from training data and user-chosen
    # choice of weights w
    def maxout_feature_transforms_testing(self,a, w):    
        # loop through each layer matrix
        c=0
        for W1,W2 in w:
            #  pad with ones (to compactly take care of bias) for next layer computation        
            o = np.ones((1,np.shape(a)[1]))
            a = np.vstack((o,a))

            # compute linear combination of current layer units
            a1 = np.dot(a.T, W1).T
            a2 = np.dot(a.T, W2).T

            # pass through activation
            a = self.activation(a1,a2)

            # get normalizer for this layer tuned to training data
            normalizer = self.normalizers[c]
            a = normalizer(a)
            c+=1
        return a