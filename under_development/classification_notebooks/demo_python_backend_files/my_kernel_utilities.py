import math
import numpy as np
import csv
from sklearn import preprocessing

# make polynomial kernel using input data
def make_poly_kernel(X,D):
    # just use scikit-learn's preprocessing for this - faster
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    K = np.zeros((P,P))
    for p in range(0,P):
        x_p = X[p,:]
        for q in range(p,P):
            x_q = X[q,:]
            K[p][q] = (np.dot(x_p,x_q.T) + 1)**D
            if q > p:
                K[q][p] = K[p][q]
            
    #temp = np.ones((np.shape(K)[0],1))
    #K = np.concatenate((temp,K),axis = 1)
    np.savetxt('poly_kernel_matrix.txt', K)
    return K
    
# use your polynomial kernel matrix to produce poly kernel features for input data
def make_poly_features(orig_X,new_X,D):
    P_o,N_o = np.shape(orig_X)
    P,N = np.shape(new_X)  # of course N = 2 here, just for visualization
    H = np.zeros((P,P_o))
    for p in range(0,P):
        x_p = new_X[p,:]
        x_p.shape = (len(x_p),1)
        temp = (np.dot(orig_X,x_p) + 1)**D
        temp.shape = (1,len(temp))
        H[p,:] = temp
    return H

# make fourier kernel using input data
def make_fourier_kernel(X,D):
    # just use scikit-learn's preprocessing for this - faster
    P,N = np.shape(X)  # of course N = 2 here, just for visualization
    K = np.zeros((P,P))

    # loop over data matrix
    for p in range(0,P):
        x_p = X[p,:]
        for q in range(p,Q):
            x_q = X[q,:]
            temp = 1
            for n in range(0,N):
                b = np.sin((2*D + 1)*np.pi*(x_p[n] - x_q[n]))
                t = sin(np.pi*(x_p[n]-x_q[n]))
                temp*=b/t
                        
            K[p,q] = temp
            if q > p:
                K[q][p] = K[p][q]
            
    np.savetxt('fourier_kernel_matrix.txt', K)
    return K        

# create kernels
def create_kernel(X,D,feat_type):
    K = 0
    # make desired feature type
    if feat_type == 'poly':
        K = make_poly_kernel(X,D)
    if feat_type == 'fourier':
        K = make_fourier_kernel(X,D)    
        
    return K
                        
# create features for classification of new points                        
def create_features(orig_X,new_X,D,feat_type):
    F = 0
    # make desired feature type
    if feat_type == 'poly':
        F = make_poly_features(orig_X,new_X,D)
    if feat_type == 'fourier':
        F = make_fourier_features(orig_X,new_X,D,feat_type)

     
    return F