# short list of utilites for calculating combinations etc., number of basis features
# as well as the features themselves
import math
import numpy as np
import csv
from sklearn import preprocessing

def poly_features(X,D):
    # just use scikit-learn's preprocessing for this - faster
    polyfit = preprocessing.PolynomialFeatures(D)
    F = polyfit.fit_transform(X)
   
    return F
    
def fourier_features(X,D):
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = (D+1)**2 + 1
    F = np.zeros((P,M))
    
    # loop over dataset, transforming each input data to fourier feature
    for p in range(0,P):
        f = np.zeros((1,M))
        x = X[p,:]
        m = 0
        
        # enumerate all individual Fourier terms - probably enumerating in complex exponential form is best. 
        for i in range(0,D+1):
            for j in range(0,D+1):
                F[p][m] = np.cos(i*x[0])*np.sin(j*x[1])
                m+=1
    return F
        

# generate and save parameters for random features
def make_random_params(X,D):
    # sizes of things
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = D
    
    # make random projections and save them
    R = np.random.randn(M,N+1)
    np.savetxt('random_projections.txt', R)    

# generate random features
def random_features(X,D):
    # create random projections for feature transformations
    R = np.loadtxt('random_projections.txt')
    
    # sizes of things
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    M = D
    F = np.zeros((P,M+1))  

    # set external biases, then tranform random projection of each point
    for p in range(0,P):   
        F[p,0] = 1
        x_p = X[p,:]
        for m in range(0,M):
            r = R[m,:]
            
            # compute random projection of x_p onto r
            proj = r[0] + np.sum(x_p*r[1:])
            
            # take nonlinear transformation of random projection
            # using cosine right now
            F[p,m+1] = np.cos(proj)
            
    return F


def create_features(X,D,feat_type):
    F = 0
    # make desired feature type
    if feat_type == 'poly':
        F = poly_features(X,D)
    if feat_type == 'fourier':
        F = fourier_features(X,D)
    if feat_type == 'random': 
        F = random_features(X,D)
     
    return F