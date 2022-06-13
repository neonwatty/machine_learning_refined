import autograd.numpy as np

class Setup:
    def __init__(self,x,name):
        normalizer = 0
        inverse_normalizer = 0
        if name == 'standard':
            # create normalizer
            self.normalizer, self.inverse_normalizer = self.standard_normalizer(x)
            
        elif name == 'sphere':
            # create normalizer
            self.normalizer, self.inverse_normalizer = self.PCA_sphereing(x)
        else:
            self.normalizer = lambda data: data
            self.inverse_normalizer = lambda data: data
 
    # standard normalization function - with nan checker / filler in-er
    def standard_normalizer(self,x):    
        # compute the mean and standard deviation of the input
        x_means = np.nanmean(x,axis = 1)[:,np.newaxis]
        x_stds = np.nanstd(x,axis = 1)[:,np.newaxis]   

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # fill in any nan values with means 
        ind = np.argwhere(np.isnan(x) == True)
        for i in ind:
            x[i[0],i[1]] = x_means[i[0]]

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    # compute eigendecomposition of data covariance matrix for PCA transformation
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        d,V = np.linalg.eigh(Cov)
        return d,V

    # PCA-sphereing - use PCA to normalize input features
    def PCA_sphereing(self,x,**kwargs):
        # Step 1: mean-center the data
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_centered = x - x_means

        # Step 2: compute pca transform on mean-centered data
        d,V = self.PCA(x_centered,**kwargs)

        # Step 3: divide off standard deviation of each (transformed) input, 
        # which are equal to the returned eigenvalues in 'd'.  
        stds = (d[:,np.newaxis])**(0.5)
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((stds.shape))
            adjust[ind] = 1.0
            stds += adjust
        
        normalizer = lambda data: np.dot(V.T,data - x_means)/stds

        # create inverse normalizer
        inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

        # return normalizer 
        return normalizer,inverse_normalizer