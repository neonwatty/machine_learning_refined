import autograd.numpy as np
import copy

class Setup:
    def __init__(self,x,y,**kwargs):        
        # create splits, levels, and dims
        self.splits,self.levels,self.dims =  self.create_boost_stumps(x,y)

        # define initializer
        self.num_classifiers = 1
        if 'num_classifiers' in kwargs:
            self.num_classifiers = kwargs['num_classifiers']
        self.scale = 0.1
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    # create initial weights for arbitrary feedforward network
    def initializer(self):
        w_init = np.zeros((len(self.splits)+1,self.num_classifiers));
        return w_init
    
    # compute transformation on entire set of inputs
    def feature_transforms(self,x): 
        # container for stump transformed data
        N = x.shape[0]
        P = x.shape[1]
        S = len(self.splits)
        x_transformed = np.zeros((S,P))

        # loop over points and transform each individually
        for pt in range(P):
            x_n = x[:,pt]

            # loop over the stump collectionand calculate weighted contribution
            for u in range(len(self.splits)):
                # get current stump f_u
                split = self.splits[u]
                level = self.levels[u]
                dim = self.dims[u]

                ### our stump function f_u(x)
                if x_n[dim] <= split:  # lies to the left - so evaluate at left level
                    x_transformed[u][pt] = level[0]
                else:
                    x_transformed[u][pt]  = level[1]
        return x_transformed

    def create_boost_stumps(self,x,y):
        '''
        Create stumps tailored to an input dataset (x,y) based on the naive method of creating
        a split point between each pair of successive inputs.  

        The input to this function: a dataset (x,y) where the input x has shape 
        (NUMBER OF POINTS by  DIMENSION OF INPUT)

        The output of this function is a set of two lists, one containing the split points and 
        the other the corresponding levels of stumps.
        '''

        # containers for the split points and levels of our stumps, along with container
        # for which dimension the stump is defined along
        splits = []
        levels = []
        dims = []

        # important constants: dimension of input N and total number of points P
        N = np.shape(x)[0]              
        P = np.size(y)

        ### begin outer loop - loop over each dimension of the input
        for n in range(N):
            # make a copy of the n^th dimension of the input data (we will sort after this)
            x_n = copy.deepcopy(x[n,:])
            y_n = copy.deepcopy(y)

            # sort x_n and y_n according to ascending order in x_n
            sorted_inds = np.argsort(x_n,axis = 0)
            x_n = x_n[sorted_inds]
            y_n = y_n[:,sorted_inds]

            # loop over points and create stump in between each 
            # in dimension n
            for p in range(P - 1):
                # compute split point
                split = (x_n[p] + x_n[p+1])/float(2)

                ### create non-zero stump to 'left' of split ###
                # compute and store split point
                splits.append(split)
                levels.append([1,0])
                dims.append(n)

                ### create non-zero stump to 'right' of split ###
                # compute and store split point
                splits.append(split)
                levels.append([0,1])
                dims.append(n)

        # return items
        return splits,levels,dims