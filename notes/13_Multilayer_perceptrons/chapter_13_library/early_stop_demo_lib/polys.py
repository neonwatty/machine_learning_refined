import autograd.numpy as np
import copy
import itertools

class Setup:
    def __init__(self,x,y,**kwargs):    
        # get desired degree
        self.D = kwargs['degree']
        self.N = x.shape[0]
        
        # all monomial terms degrees
        degs = np.array(list(itertools.product(list(np.arange(self.D+1)), repeat = self.N)))
        b = np.sum(degs,axis = 1)
        ind = np.argwhere(b <= self.D)
        ind = [v[0] for v in ind]
        degs = degs[ind,:]     
        self.degs = degs[1:,:]

        # define initializer
        self.num_classifiers = 1
        if 'num_classifiers' in kwargs:
            self.num_classifiers = kwargs['num_classifiers']
        self.scale = 0.1
        if 'scale' in kwargs:
            self.scale = kwargs['scale']

    # create initial weights for arbitrary feedforward network
    def initializer(self):
        w_init = self.scale*np.random.randn(len(self.degs)+1,self.num_classifiers);
        return w_init
    
    # compute transformation on entire set of inputs
    def feature_transforms(self,x): 
        x_transformed = np.array([np.prod(x**v[:,np.newaxis],axis = 0)[:,np.newaxis] for v in self.degs])[:,:,0]  
        return x_transformed