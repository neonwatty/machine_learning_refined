import autograd.numpy as np

class My_Normalizers:
    '''
    A class that wraps up the various input normalization schemes
    we have seen including
    - mean centering / std normalization
    - PCA sphereing
    
    For each scheme you put in input features, and the following is returned
    - normalizer: the normalization scheme of your choice, returned as a function that 
    you can then use for future test points
    
    You can then normalize the input x of a dataset using the desired normalization scheme
    by 
    
    x_normalized = normalizer(x)
    '''    

    ###### standard normalization function ######
    def standard(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # return normalizer 
        return normalizer

    ###### standard normalization function ######
    # compute eigendecomposition of data covariance matrix
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        D,V = np.linalg.eigh(Cov)
        return D,V

    # PCA-sphereing - use PCA to normalize input features
    def PCA_sphereing(self,x,**kwargs):
        # standard normalize the input data
        standard_normalizer = self.standard(x)
        x_standard = standard_normalizer(x)
        
        # compute pca transform 
        D,V = self.PCA(x_standard,**kwargs)
        
        # compute forward sphereing transform
        D_ = np.array([1/d**(0.5) for d in D])
        D_ = np.diag(D_)
        W = np.dot(D_,V.T)
        pca_sphere_normalizer = lambda data: np.dot(W,standard_normalizer(data))

        # return normalizer 
        return pca_sphere_normalizer
