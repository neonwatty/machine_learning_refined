import numpy as np
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import math
import time

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

######### centering and contrast-normalizing #########
def center(X):
    '''
    A function for normalizing each feaure dimension of an input array, mean-centering
    and division by its standard deviation
    
    '''
    X_means = np.mean(X,axis=0)[np.newaxis,:]
    X_normalized = X - X_means

    return X_normalized

def contrast_normalize(X):
    '''
    A contrast-normalizing function for image data pre-sphereing normalization.
    
    '''
    # compute and subtract off means
    X_means = np.mean(X,axis=0)[np.newaxis,:]
    X = X - X_means
    
    # divide off std of each image - remove any images deemed constant (whose std = 0)
    X_stds = np.std(X,axis=0)[np.newaxis,:]
    ind = np.argwhere(np.abs(X_stds) > 10**(-7))
    ind = np.array([s[1] for s in ind])
    X = X[:,ind]
    X_stds = X_stds[:,ind]
    X_normalized = X/X_stds
    
    # print report to user if any patches deemed constant
    report = np.shape(X_means)[1] - len(ind) 
    if report > 0:
        print (str(report) + ' images of ' + str(np.shape(X_means)[1]) + ' imagses found to be constant, and so were removed')

    return X_normalized

########## sphereing pre-processing functionality ##########
def compute_pcs(X,lam):
    '''
    A function for computing the principal components of an input data matrix.  Both
    principal components and variance parameters (eigenvectors and eigenvalues of XX^T)
    are returned
    '''
    # create the correlation matrix
    P = float(X.shape[1])
    Cov = 1/P*np.dot(X,X.T) + lam*np.eye(X.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    D,V = np.linalg.eigh(Cov)
    return V, D

def pca_transform_data(X,**kwargs):
    '''
    A function for producing the full PCA transformation on an input dataset X.  
    '''
    # user-determined number of principal components to keep, and regularizer penalty param
    num_components = X.shape[0]
    if 'num_components' in kwargs:
        num_components = kwargs['num_components']
    lam = 10**(-7)
    if 'lam' in kwargs:
        lam = kwargs['lam']
    
    # compute principal components
    V,D = compute_pcs(X,lam)
    V = V[:,-num_components:]
    D = D[-num_components:]

    # compute transformed data for PC space: V^T X
    W = np.dot(V.T,X)
    return W,V,D
      
def PCA_sphere(X,**kwargs):
    '''
    A function for producing the full PCA sphereing on an input dataset X.  
    '''
    # compute principal components
    W,V,D = pca_transform_data(X,**kwargs)
    
    # compute transformed data for PC space: V^T X
    W = np.dot(V.T,X)
    D_ = np.array([1/d**(0.5) for d in D])
    D_ = np.diag(D_)
    S = np.dot(D_,W)
    return W,S

def ZCA_sphere(X,**kwargs):
    '''
    A function for producing the full PCA sphereing on an input dataset X.  
    '''   
    
    # compute principal components
    W,V,D = pca_transform_data(X,**kwargs)
    
    # PCA-sphere data
    W = np.dot(V.T,X)
    D_ = np.array([1/d**(0.5) for d in D])
    D_ = np.diag(D_)
    S = np.dot(D_,W)
    
    # rotate data back to original orientation - ZCA sphere
    Z = np.dot(V,S)
    
    return W,S,Z

########## plotting functionality ############
def show_images(X):
    '''
    Function for plotting input images, stacked in columns of input X.
    '''
    # plotting mechanism taken from excellent answer from stack overflow: https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    plt.figure(figsize = (9,3))
    gs1 = gridspec.GridSpec(5, 14)
    gs1.update(wspace=0, hspace=0.05) # set the spacing between axes. 
    
    # shape of square version of image
    square_shape = int((X.shape[0])**(0.5))

    for i in range(min(70,X.shape[1])):
        # plot image in panel
        ax = plt.subplot(gs1[i])
        im = ax.imshow(255 - np.reshape(X[:,i],(square_shape,square_shape)),cmap = 'gray')

        # clean up panel
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()